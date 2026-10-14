"""Synthetic-only reporting lineage/accounting tests; no evaluation is opened."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest import mock
import make_report as mr


class ReportAdditionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.here = self.root / 'research/bihash_future_prediction_20261013'
        self.here.mkdir(parents=True)
        self.patch = mock.patch.multiple(mr, ROOT=self.root, HERE=self.here, CACHE=self.root/'.cache/bihash_future_prediction_20261013')
        self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    def file(self, name, value):
        path = self.here / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': mr.sha(path)}

    def merged(self):
        producers, cases, ae = [], [], []
        failure = {'stage': 'data_prepare', 'dataset': 'uci235_household_hourly', 'error': 'raw file missing', 'status': 'FAILED_PATH_PRESERVED'}
        for tag, dataset, status, wall in (('fresh_v1','uci274_sml2010','FAILED_PRESERVED',10.),('household_recovery_v1','uci235_household_hourly','COMPLETE',20.)):
            case = [{'dataset': dataset, 'seed': i%3, 'label': f'case{i}'} for i in range(36)]
            pretrain = [{'dataset': dataset, 'seed': i} for i in range(3)]
            receipt = self.file('resources/'+tag+'.json', {'tag':tag,'status':status,'gpu':True,'wall_seconds':wall,'finished_unix':2.})
            value = {'tag':tag,'status':status,'cases':case,'future_pretrains':pretrain,'failed_paths':[failure] if status!='COMPLETE' else [],'resource_receipt':receipt}
            if status=='COMPLETE': value['data_repair_result']=self.file('repair.json',{'tag':'repair','status':'COMPLETE','phase':'ingest_repair','wall_seconds':1.5})
            reference = self.file('run_'+tag+'/RESULT.json',value)
            snapshot = self.file('snapshot_'+tag+'.json',{'sha256':{'source.py':'a'*64}})
            producers.append({'result':reference,'tag':tag,'original_status':status,'valid_case_count':36,'valid_future_AE_count':3,'datasets':[dataset],
                              'source_snapshot_manifest':snapshot,'resource_receipt':receipt,'original_failed_paths':value['failed_paths']})
            for original in case:
                copied=copy.deepcopy(original)
                copied['producer_lineage']={'producer_result':reference,'original_status':status,'producer_tag':tag,
                    'original_case_json':self.file('run_'+tag+'/'+dataset+'/'+original['label']+'.json',original)}
                cases.append(copied)
            for original in pretrain:
                copied=copy.deepcopy(original); copied['producer_lineage']={'producer_result':reference,'original_status':status,'producer_tag':tag}; ae.append(copied)
        merge_receipt=self.file('resources/merge.json',{'tag':'merge','status':'COMPLETE','gpu':False,'wall_seconds':2.,'finished_unix':3.})
        return {'status':'COMPLETE','stage':'fresh','merge_provenance':{'producer_result_count':2},'producers':producers,'cases':cases,'future_pretrains':ae,
                'failed_paths':[],'recovered_failed_paths':[{**failure,'producer_result':producers[0]['result']}],
                'resource_receipt':merge_receipt,'original_gpu_wall_seconds_counted_once':30.,'new_training_updates_in_merge':0}

    def test_recovered_complete_retains_original_failed_status_and_reused_training(self):
        evidence=mr.recovery_evidence(self.merged())
        text='\n'.join(mr.recovery_text(evidence))
        self.assertIn('FAILED_PRESERVED',text); self.assertIn('SML',text)
        self.assertIn('재학습',text); self.assertIn('CPU',text)
        self.assertEqual(len(evidence['case_lineage']),72)
        self.assertEqual(len(evidence['recovered_failures']),1)
        self.assertEqual(evidence['data_repairs'][0]['status'],'COMPLETE')

    def test_resource_receipts_and_summary_are_not_double_counted(self):
        merged=self.merged()
        rows=mr.resources_from_inputs({'fresh':merged})
        self.assertEqual(len(rows),3)
        self.assertEqual(sum(r['wall_seconds'] for r in rows if r['gpu']),30.)
        self.assertEqual(sum(r['wall_seconds'] for r in rows if not r['gpu']),2.)

    def test_missing_merged_lineage_is_explicitly_unavailable(self):
        evidence=mr.recovery_evidence({'status':'FAILED_PRESERVED','failed_paths':[]})
        self.assertEqual(evidence['status'],'MERGED_LINEAGE_NOT_AVAILABLE')
        self.assertIn('미제공','\n'.join(mr.recovery_text(evidence)))

    def test_wrong_producer_SHA_is_refused(self):
        merged=self.merged(); merged['producers'][0]['result']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'RECOVERY.*SHA'): mr.recovery_evidence(merged)

    def query_notes(self):
        folder=self.here/'decision_checks_v1/hash_query_capacity_v1'
        for tag,status,process,main in (('selected_maps_v1','PARTIAL_OR_FAILED_DIAGNOSTIC_PRESERVED',1.489106178,0.15409),('selected_maps_v2','COMPLETE_AVAILABLE_CASE_DIAGNOSTIC',3.004011869,1.707143)):
            value={'status':status,'conditions':[],'required_conditions_read':18 if tag.endswith('v2') else 0,'optional_Household_conditions_read':3 if tag.endswith('v2') else 0,
                   'resource_receipt':{'gpu':False,'process_wall_seconds_including_imports':process,'main_wall_seconds':main},
                   'raw_dataset_files_opened':0,'EVAL_target_opened':False,'GPU_executed':False}
            self.file('decision_checks_v1/hash_query_capacity_v1/'+tag+'/RESULT.json',value)
        return mr.decision_notes(SimpleNamespace())

    def test_actual_query_notes_keep_limits_and_failed_attempt(self):
        notes=self.query_notes(); text='\n'.join(mr.decision_text(notes))
        for phrase in ('33','synthetic','Gumbel','고정','실패 원인','selected_maps_v1'):
            self.assertIn(phrase,text)
        self.assertIn('18/18',text); self.assertIn('3/6',text)

    def test_missing_optional_query_is_not_invented(self):
        notes=mr.decision_notes(SimpleNamespace())
        self.assertNotIn('hash_query_capacity',notes)
        self.assertIn('제공되지','\n'.join(mr.decision_text(notes)))

    def test_query_CPU_process_receipts_count_both_attempts_once_not_main_again(self):
        notes=self.query_notes(); notes['same_success_again']=notes['hash_query_capacity']
        audit=self.file('audit.json',{'status':'PASS'})
        rows=mr.cpu_accounting([],notes,self.root/audit['path'])
        total=sum(r['counted_wall_seconds'] for r in rows if r['counted_wall_seconds'] is not None)
        self.assertAlmostEqual(total,1.489106178+3.004011869)

    def test_final_override_retains_current_and_labels_roles(self):
        current=self.file('decision_checks_v1/preservation_current_v1/RESULT.json',{'status':'PASS','checked_artifacts':1,'expected_artifacts':1})
        final=self.file('decision_checks_v1/preservation_final_v1/RESULT.json',{'status':'PASS','checked_artifacts':1,'expected_artifacts':1})
        notes=mr.decision_notes(SimpleNamespace(preservation_current_result=self.root/final['path']))
        paths={str(n['path']) for n in notes.values()}
        self.assertIn(str(self.root/current['path']),paths)
        self.assertEqual(len(paths),len(notes))
        text='\n'.join(mr.decision_text(notes))
        self.assertIn('중간 시점',text); self.assertIn('최종 역할',text)


if __name__=='__main__': unittest.main()
