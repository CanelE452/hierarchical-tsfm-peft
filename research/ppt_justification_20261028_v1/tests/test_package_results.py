"""Synthetic CPU fixtures only; no campaign results or GPU are accessed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from research.ppt_justification_20261028_v1 import package_results as pack


class ResultsBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name)
        for name in pack.REQUIRED:
            self.put(pack.MAIN+name,{} if name.endswith('.json') else 'synthetic output')
        for name in pack.SOURCE_NAMES:self.put('research/'+pack.NAME+'/'+name,'# synthetic source\n')
        self.source='research/'+pack.NAME+'/models.py'
        self.put(pack.MAIN+'DELIVERY_SCOPE.json',{'status':'USER_ASSIGNED_EXPERIMENTS_AND_EVIDENCE_ONLY'})
        self.put(pack.MAIN+'PREREGISTRATION.json',{'registered_sources':{self.source:self.digest(self.source)}})
        self.plan={'jobs':[{'job_id':f'base_{i}'} for i in range(20)],
                   'optional_jobs':[{'job_id':f'optional_{i}'} for i in range(10)],
                   'extension_jobs':[{'job_id':f'extension_{i}'} for i in range(30)]}
        self.put(pack.MAIN+'RUN_PLAN.json',self.plan)
        self.put(pack.MAIN+'REGISTRATION_SEAL.json',{'files':{n:self.digest(pack.MAIN+n) for n in
            ('PREREGISTRATION.json','RUN_PLAN.json')}})
        statuses={j['job_id']:('COMPLETE' if family=='jobs' else 'NOT_RUN_BUDGET')
                  for family,jobs in self.plan.items() for j in jobs}
        self.completion={'status':'TRAINING_COMPLETE','active':None,'base_complete_count':20,'run_statuses':statuses}
        self.put(pack.MAIN+'EXECUTION_COMPLETION.json',self.completion)
        self.analysis={'status':'PASS','base_complete':20,'reference_rows':20,'T_numeric_values_parsed':0,
            'new_result_count':20,'optional_complete':0,'coverage':[{'synthetic':True}],
            'inputs':{'completion_sha256':self.digest(pack.MAIN+'EXECUTION_COMPLETION.json'),
                      'registration_seal_sha256':self.digest(pack.MAIN+'REGISTRATION_SEAL.json')},
            'verified_files':{pack.MAIN+n:self.digest(pack.MAIN+n) for n in pack.OUTPUTS}}
        self.put(pack.MAIN+'ANALYSIS_VALIDATION.json',self.analysis)
        e_path=pack.E+'RESULTS_DISPOSITION.json';cost_path=pack.COST+'RESULTS_DISPOSITION.json'
        self.put(e_path,{'status':'NOT_RUN_BUDGET','reason':'Synthetic deadline budget fixture'})
        self.put(cost_path,{'status':'NOT_MEASURED','phase':'BUDGET','reason':'Synthetic measurement not admitted'})
        reviewed=[pack.MAIN+n for n in pack.OUTPUTS+('ANALYSIS_VALIDATION.json','EXECUTION_COMPLETION.json','DELIVERY_SCOPE.json')]
        reviewed += [pack.MAIN+n+'.png' for n in pack.FIGURES]+[e_path,cost_path]
        self.gate={'status':'PASS','scope':'EXPERIMENT_RESULTS_ONLY','figures_visual_review':'PASS',
            'delivery_scope_sha256':self.digest(pack.MAIN+'DELIVERY_SCOPE.json'),
            'main':{'analysis_sha256':self.digest(pack.MAIN+'ANALYSIS_VALIDATION.json'),
                    'completion_sha256':self.digest(pack.MAIN+'EXECUTION_COMPLETION.json')},
            'e':{'status':'NOT_RUN_BUDGET','disposition_path':e_path,'disposition_sha256':self.digest(e_path)},
            'cost':{'status':'NOT_MEASURED','disposition_path':cost_path,'disposition_sha256':self.digest(cost_path)},
            'reviewed_files':{p:self.digest(p) for p in reviewed}}
        self.save_gate()

    def put(self,relative,value):
        p=self.repo/relative;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value,encoding='utf8')
        return p

    def digest(self,relative):return pack.sha(self.repo/relative)

    def save_gate(self):self.put(pack.MAIN+'ROOT_RESULTS_GATE.json',self.gate)

    def refresh_main(self):
        self.put(pack.MAIN+'EXECUTION_COMPLETION.json',self.completion)
        self.analysis['inputs']['completion_sha256']=self.digest(pack.MAIN+'EXECUTION_COMPLETION.json')
        self.put(pack.MAIN+'ANALYSIS_VALIDATION.json',self.analysis)
        for filename,key in [('ANALYSIS_VALIDATION.json','analysis_sha256'),('EXECUTION_COMPLETION.json','completion_sha256')]:
            self.gate['main'][key]=self.digest(pack.MAIN+filename)
            self.gate['reviewed_files'][pack.MAIN+filename]=self.digest(pack.MAIN+filename)
        self.save_gate()

    def include_e(self,provisional=False):
        completion={'status':'TRAINING_COMPLETE','active':None}
        self.put(pack.E+'EXECUTION_COMPLETION.json',completion)
        ev={'status':'PASS','T_numeric_values_parsed':0,'base_complete':5,'primary_complete':True,
            'inputs':{'completion_sha256':self.digest(pack.E+'EXECUTION_COMPLETION.json')},
            'verified_files':{self.source:self.digest(self.source)}}
        self.put(pack.E+'COMPLETION_VALIDATION.json',ev)
        names=('RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv','RESULTS_KO.md',
               'FIG_E_FIXED.png','FIG_E_FIXED.svg','FIG_E_FIXED.pdf')
        for name in names:self.put(pack.E+name,'synthetic E output')
        ea={'status':'PASS','base_complete':5,'reference_rows':5,'T_numeric_values_parsed':0,
            'coverage':{'primary_n':5,'primary_complete':True,'scientific_complete':not provisional,'provisional':provisional},
            'summary':[{'method_id':'OURS','n':5,'seed_ids':json.dumps(list(range(92601,92606))),'provisional':False},
                       {'method_id':'E_FIXED','n':5,'seed_ids':json.dumps(list(range(92601,92606))),'provisional':provisional}],
            'inputs':{'completion_sha256':self.digest(pack.E+'EXECUTION_COMPLETION.json'),
                      'completion_validation_sha256':self.digest(pack.E+'COMPLETION_VALIDATION.json')},
            'verified_files':{pack.E+n:self.digest(pack.E+n) for n in names}}
        self.put(pack.E+'ANALYSIS_VALIDATION.json',ea)
        self.gate['e']={'status':'VERIFIED_RESULT',**{key:self.digest(pack.E+filename) for filename,key in
            [('ANALYSIS_VALIDATION.json','analysis_sha256'),('EXECUTION_COMPLETION.json','completion_sha256'),
             ('COMPLETION_VALIDATION.json','completion_validation_sha256')]}}
        for n in names+('ANALYSIS_VALIDATION.json','COMPLETION_VALIDATION.json','EXECUTION_COMPLETION.json'):
            if not n.endswith(('.svg','.pdf')):self.gate['reviewed_files'][pack.E+n]=self.digest(pack.E+n)
        self.save_gate();return ea

    def include_cost(self):
        summary={'status':'COMPLETE','training_executed':False,'T_numeric_values_parsed':0,'rows':[
            {'accuracy_run_id':n} for n in ('Raw_appliances','KTAU_main_appliances_K10_s92601','KTAU_main_appliances_K10_s92602')]}
        self.put(pack.COST+'COST_SUMMARY.json',summary)
        validation={'status':'PASS','training_executed':False,'T_numeric_values_parsed':0,
            'cost_summary_sha256':self.digest(pack.COST+'COST_SUMMARY.json'),
            'evidence_bindings':{'COST_SUMMARY.json':self.digest(pack.COST+'COST_SUMMARY.json')},
            'producer_source_bindings':{self.source:self.digest(self.source)}}
        self.put(pack.COST+'COST_VALIDATION.json',validation)
        self.gate['cost']={'status':'VERIFIED','summary_sha256':self.digest(pack.COST+'COST_SUMMARY.json'),
                           'validation_sha256':self.digest(pack.COST+'COST_VALIDATION.json')}
        for n in ('COST_SUMMARY.json','COST_VALIDATION.json'):self.gate['reviewed_files'][pack.COST+n]=self.digest(pack.COST+n)
        self.save_gate()

    def test_honest_exclusions_need_no_deck_or_presentation_gate(self):
        files,bindings,metadata=pack.prepare_results(self.repo)
        self.assertEqual(metadata['e_status'],'NOT_RUN_BUDGET')
        self.assertEqual(metadata['cost_status'],'NOT_MEASURED')
        self.assertTrue(all('presentation' not in p.parts for p in files))

    def test_decks_drafts_raw_arrays_and_unknown_jobs_are_not_collected(self):
        denied=[pack.MAIN+'presentation/BUILD_FINAL.json',pack.MAIN+'presentation/deck.pptx',
            pack.MAIN+'ROOT_PRESENTATION_GATE.json',pack.MAIN+'SPEAKER_NOTES_KO.md',pack.MAIN+'talk.pdf',
            pack.COST+'COST_PROTOCOL_v3_DRAFT.json',pack.COST+'private_identity.json',
            '.cache/'+pack.NAME+'/predictions.npz',pack.MAIN+'checkpoint.pt',
            pack.MAIN+'jobs/unregistered/RESULT.json']
        for p in denied:self.put(p,'unusable sentinel')
        files,_,_=pack.prepare_results(self.repo)
        self.assertFalse({self.repo/p for p in denied}&files)

    def test_reviewed_scientific_figure_pdf_and_interim_audit_are_allowed(self):
        path=pack.MAIN+'INTERIM_PPT_EVIDENCE_001.json';self.put(path,{'status':'INTERIM'})
        self.gate['reviewed_files'][path]=self.digest(path);self.save_gate()
        files,_,_=pack.prepare_results(self.repo)
        self.assertIn(self.repo/path,files)
        self.assertIn(self.repo/(pack.MAIN+'FIG_F_REPLACEMENT.pdf'),files)

    def test_terminal_execution_and_twenty_required_runs_are_enforced(self):
        for change in ({'status':'RUNNING'},{'active':'base_1'},{'base_complete_count':19}):
            with self.subTest(change=change):
                old=dict(self.completion);self.completion.update(change);self.refresh_main()
                with self.assertRaises(ValueError):pack.prepare_results(self.repo)
                self.completion=old;self.refresh_main()
        self.completion['run_statuses']['optional_0']='RUNNING';self.refresh_main()
        with self.assertRaisesRegex(ValueError,'nonterminal'):pack.prepare_results(self.repo)

    def test_interim_analysis_cannot_replace_final_analysis(self):
        self.analysis['status']='INTERIM';self.refresh_main()
        with self.assertRaisesRegex(ValueError,'scientific analysis'):pack.prepare_results(self.repo)

    def test_stale_review_source_or_analysis_is_rejected(self):
        for relative in (self.source,pack.MAIN+'SUMMARY.csv',pack.MAIN+'ANALYSIS_VALIDATION.json'):
            with self.subTest(relative=relative):
                p=self.repo/relative;original=p.read_bytes();p.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError,'changed'):pack.prepare_results(self.repo)
                p.write_bytes(original)

    def test_missing_independent_visual_review_is_rejected(self):
        self.gate['figures_visual_review']='PENDING';self.save_gate()
        with self.assertRaisesRegex(ValueError,'scientific-figure'):pack.prepare_results(self.repo)

    def test_missing_reviewed_required_table_is_rejected(self):
        del self.gate['reviewed_files'][pack.MAIN+'CONTRASTS.csv'];self.save_gate()
        with self.assertRaisesRegex(ValueError,'missing required outputs'):pack.prepare_results(self.repo)

    def test_path_escape_and_private_reviewed_files_are_rejected(self):
        for name in ('../outside.json','C:/private.json',pack.MAIN+'presentation/secret.json',
                     pack.MAIN+'talk.pdf',pack.COST+'private_policy.json',pack.MAIN+'SPEAKER_NOTES.md',
                     pack.COST+'inventory.json',pack.COST+'private_inventory.json',pack.COST+'raw_logs.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):pack.public_path(self.repo,name)

    def test_complete_e_and_provisional_five_seed_result_are_both_preserved(self):
        for provisional in (False,True):
            with self.subTest(provisional=provisional):
                self.include_e(provisional)
                files,_,metadata=pack.prepare_results(self.repo)
                self.assertIn(self.repo/(pack.E+'FIG_E_FIXED.pdf'),files)
                self.assertEqual(metadata['e_coverage']['provisional'],provisional)

    def test_unfinished_e_extension_cannot_lose_provisional_label(self):
        ea=self.include_e(True);ea['coverage']['provisional']=False
        self.put(pack.E+'ANALYSIS_VALIDATION.json',ea)
        self.gate['e']['analysis_sha256']=self.digest(pack.E+'ANALYSIS_VALIDATION.json')
        self.gate['reviewed_files'][pack.E+'ANALYSIS_VALIDATION.json']=self.digest(pack.E+'ANALYSIS_VALIDATION.json');self.save_gate()
        with self.assertRaisesRegex(ValueError,'remain provisional'):pack.prepare_results(self.repo)

    def test_verified_e_cannot_be_hidden_as_budget_excluded(self):
        old=dict(self.gate['e']);self.include_e();self.gate['e']=old;self.save_gate()
        with self.assertRaisesRegex(ValueError,'cannot be omitted'):pack.prepare_results(self.repo)

    def test_e_same_size_different_seed_group_is_rejected(self):
        ea=self.include_e();ea['summary'][1]['seed_ids']=json.dumps(list(range(92602,92607)))
        self.put(pack.E+'ANALYSIS_VALIDATION.json',ea)
        self.gate['e']['analysis_sha256']=self.digest(pack.E+'ANALYSIS_VALIDATION.json')
        self.gate['reviewed_files'][pack.E+'ANALYSIS_VALIDATION.json']=self.digest(pack.E+'ANALYSIS_VALIDATION.json');self.save_gate()
        with self.assertRaisesRegex(ValueError,'five-seed groups'):pack.prepare_results(self.repo)

    def test_e_incomplete_requires_actual_matching_status_reason(self):
        p=self.gate['e']['disposition_path'];self.put(p,{'status':'INCOMPLETE','reason':'Synthetic interruption'})
        self.gate['e'].update(status='INCOMPLETE',disposition_sha256=self.digest(p))
        self.gate['reviewed_files'][p]=self.digest(p);self.save_gate()
        self.assertEqual(pack.prepare_results(self.repo)[2]['e_status'],'INCOMPLETE')

    def test_cost_verified_result_and_nonresult_are_separate(self):
        self.include_cost();files,_,metadata=pack.prepare_results(self.repo)
        self.assertEqual(metadata['cost_status'],'VERIFIED')
        self.assertIn(self.repo/(pack.COST+'COST_SUMMARY.json'),files)

    def test_operational_amendments_and_archives_flow_into_results_only_bundle(self):
        expected=set()
        for relative in (pack.COST+'AMENDMENT_003.json',pack.COST+'HOST_ADMISSION_FAILURE_001.json',pack.COST+'COST_PROTOCOL.json',
                         pack.E+'AMENDMENT_002.json'):
            expected.add(self.put(relative,{'status':'SYNTHETIC_PUBLIC_EVIDENCE'}))
        for archive,sources in (
            (pack.COST+'revisions/preflight_003/',[pack.COST+'HOST_GPU_CONTRACT.json',
                'research/ppt_justification_20261028_v1/host_gpu_gate.py',pack.COST+'COST_PROTOCOL_v3_DRAFT.json']),
            (pack.E+'revisions/preflight_002/',[pack.E+'AMENDMENT_001.json',
                pack.COST+'HOST_GPU_CONTRACT.json','research/ppt_e_fixed_20261028_v1/postprocess.py'])):
            rows=[]
            for source in sources:
                relative=archive+source;p=self.put(relative,'preserved synthetic original')
                rows.append({'source':source,'archived':relative,'bytes':p.stat().st_size,'sha256':self.digest(relative)})
                if 'DRAFT' not in source:expected.add(p)
            expected.add(self.put(archive+'ARCHIVE_RECEIPT.json',{'files':rows}))
        denied=[pack.COST+'COST_PROTOCOL_v4_DRAFT.json',pack.COST+'HOST_GPU_CONTRACT_v2_DRAFT.json',
                pack.COST+'private_inventory.json','.cache/ppt_justification_20261028_v1/host_inventory.json']
        for relative in denied:self.put(relative,'not public')
        files,bindings,metadata=pack.prepare_results(self.repo)
        self.assertTrue(expected<=files)
        self.assertFalse(any('draft' in p.name.lower() for p in files))
        self.assertFalse({self.repo/p for p in denied}&files)
        self.assertTrue(all(bindings[p.relative_to(self.repo).as_posix()]==pack.sha(p) for p in expected))
        excluded=metadata['archive_members_excluded']
        self.assertEqual(len(excluded),1)
        self.assertEqual(excluded[0]['source'],pack.COST+'COST_PROTOCOL_v3_DRAFT.json')
        self.assertEqual(excluded[0]['sha256'],self.digest(excluded[0]['path']))
        receipt=pack.write_results_bundle(self.repo,files,bindings,metadata)
        self.assertEqual(receipt['archive_members_excluded'],excluded)
        with zipfile.ZipFile(receipt['zip_path']) as archive:
            manifest=json.loads(archive.read(pack.MAIN+pack.MANIFEST_NAME))
            self.assertEqual(manifest['archive_members_excluded'],excluded)
            self.assertNotIn(excluded[0]['path'],archive.namelist())

    def test_contaminated_cost_excludes_aggregate_and_explains_phase(self):
        p=self.gate['cost']['disposition_path']
        self.put(p,{'status':'CONTAMINATED','phase':'MEASUREMENT','reason':'Synthetic host contamination'})
        self.put(pack.COST+'COST_SUMMARY.json',{'status':'UNVERIFIED'})
        self.gate['cost'].update(status='CONTAMINATED',disposition_sha256=self.digest(p))
        self.gate['reviewed_files'][p]=self.digest(p);self.save_gate()
        files,_,metadata=pack.prepare_results(self.repo)
        self.assertNotIn(self.repo/(pack.COST+'COST_SUMMARY.json'),files)
        self.assertEqual(metadata['cost_status'],'CONTAMINATED')
        self.put(p,{'status':'CONTAMINATED','reason':'Phase omitted'})
        self.gate['cost']['disposition_sha256']=self.digest(p);self.gate['reviewed_files'][p]=self.digest(p);self.save_gate()
        with self.assertRaisesRegex(ValueError,'explicit phase'):pack.prepare_results(self.repo)

    def test_verified_cost_cannot_be_silently_omitted(self):
        old=dict(self.gate['cost']);self.include_cost();self.gate['cost']=old;self.save_gate()
        with self.assertRaisesRegex(ValueError,'silently excluded'):pack.prepare_results(self.repo)

    def test_incomplete_cost_facet_set_cannot_be_called_verified(self):
        self.include_cost();summary=pack.read(self.repo/(pack.COST+'COST_SUMMARY.json'));summary['rows'].pop()
        self.put(pack.COST+'COST_SUMMARY.json',summary)
        validation=pack.read(self.repo/(pack.COST+'COST_VALIDATION.json'))
        validation['cost_summary_sha256']=self.digest(pack.COST+'COST_SUMMARY.json')
        validation['evidence_bindings']['COST_SUMMARY.json']=validation['cost_summary_sha256']
        self.put(pack.COST+'COST_VALIDATION.json',validation)
        self.gate['cost'].update(summary_sha256=self.digest(pack.COST+'COST_SUMMARY.json'),validation_sha256=self.digest(pack.COST+'COST_VALIDATION.json'))
        for n in ('COST_SUMMARY.json','COST_VALIDATION.json'):self.gate['reviewed_files'][pack.COST+n]=self.digest(pack.COST+n)
        self.save_gate()
        with self.assertRaisesRegex(ValueError,'three registered facets'):pack.prepare_results(self.repo)

    def test_synthetic_zip_has_exact_members_crc_manifest_and_all_hashes(self):
        files,bindings,metadata=pack.prepare_results(self.repo)
        old=self.put(pack.MAIN+pack.NAME+'_bundle.zip','existing old bundle')
        receipt=pack.write_results_bundle(self.repo,files,bindings,metadata)
        self.assertEqual(old.read_text(),'existing old bundle')
        with zipfile.ZipFile(receipt['zip_path']) as z:
            self.assertIsNone(z.testzip())
            manifest=json.loads(z.read(pack.MAIN+pack.MANIFEST_NAME))
            expected={p.relative_to(self.repo).as_posix() for p in files}|{pack.MAIN+pack.MANIFEST_NAME}
            self.assertEqual(set(z.namelist()),expected)
            self.assertEqual(z.read(pack.MAIN+pack.MANIFEST_NAME),(self.repo/(pack.MAIN+pack.MANIFEST_NAME)).read_bytes())
            for row in manifest['files']:
                import hashlib
                self.assertEqual(hashlib.sha256(z.read(row['path'])).hexdigest(),row['sha256'])
        self.assertEqual(receipt['manifest_hashes'],'PASS')
        with self.assertRaises(FileExistsError):pack.write_results_bundle(self.repo,files,bindings,metadata)

    def test_postgate_mutation_or_zip_corruption_cannot_publish(self):
        files,bindings,metadata=pack.prepare_results(self.repo)
        p=self.repo/(pack.MAIN+'SUMMARY.csv');p.write_text('mutated')
        with self.assertRaisesRegex(ValueError,'changed before packaging'):pack.write_results_bundle(self.repo,files,bindings,metadata)
        self.assertFalse((self.repo/(pack.MAIN+pack.NAME+'_results_bundle.zip')).exists())
        p.write_text('synthetic output',encoding='utf8')
        with patch.object(zipfile.ZipFile,'testzip',return_value='corrupt_member'):
            with self.assertRaisesRegex(ValueError,'CRC'):pack.write_results_bundle(self.repo,files,bindings,metadata)
        self.assertFalse((self.repo/(pack.MAIN+pack.NAME+'_results_bundle.zip')).exists())
        self.assertFalse(list((self.repo/pack.MAIN).glob('*.zip.tmp')))


if __name__=='__main__':unittest.main()
