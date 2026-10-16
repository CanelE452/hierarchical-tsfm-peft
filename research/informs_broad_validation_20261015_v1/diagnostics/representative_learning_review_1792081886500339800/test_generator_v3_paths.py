"""Explicit synthetic metadata regression tests; no arrays, models or GPU."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location('learning_'+name,HERE/name)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class CanonicalVPaths(unittest.TestCase):
    def setUp(self):
        self.generator=module(os.environ.get('LEARNING_GENERATOR_TEST_SOURCE','generate_v3.py'))
        self.temp=tempfile.TemporaryDirectory(prefix='learning_path_fixture_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        sealed=json.loads((HERE/'snapshot_1792082073095310300/INPUT_BUNDLE.json').read_text(encoding='utf-8'))
        original=sealed['run_ids'][0]
        ext=json.loads(sealed['files']['runs/'+original+'/LEARNING_EXTENSION_RECEIPT.json']['content_utf8'])
        old_parent=ext['extension_provenance']['identity']['parent_run_id']
        self.parent='primary_beijing_E4_ORDERED_MULTI_K_k11_s92601'
        self.identity=self.parent+'__e3'
        replacements=lambda value:value.replace(original,self.identity).replace(old_parent,self.parent)
        needed=['runs/'+original+'/BASE_FIT_RECEIPT.json','runs/'+original+'/LEARNING_EXTENSION_RECEIPT.json',
                'runs/'+original+'/search/CURVE.csv','runs/'+old_parent+'/BASE_FIT_RECEIPT.json']
        for relative in needed:
            text=replacements(sealed['files'][relative]['content_utf8'])
            if relative.endswith('.json'):
                parsed=json.loads(text)
                if 'family' in parsed:parsed['family']='B0'
                text=json.dumps(parsed)
            self.put(replacements(relative),text)
        self.manifests={}
        for old,new in ((old_parent,self.parent),(original,self.identity)):
            manifest=json.loads(replacements(sealed['files']['predictions/'+old+'/V_select/PREDICTION_MANIFEST.json']['content_utf8']))
            meta=manifest['metadata'];meta['parent_artifact_sha256']=meta.pop('artifact_sha256')
            meta.pop('reused_prediction_cache',None)
            meta.update(parent_run_id=new,variant='shared',inference_latent_k=11,stored_latent_k=11)
            self.manifests[new]=manifest
            self.put('ordered_predictions/'+new+'/PREDICTION_MANIFEST.json',json.dumps(manifest))
            self.put('runs/'+new+'/SOURCE_CONTRACT.json',json.dumps(dict(data_sha=manifest['data_sha'],spec=dict(
                run_id=new,family='B0',latent_k=11,k_grid=[3,6,11],fold='primary',seed=92601))))
        fields=['experiment_id','dataset','panel','fold','block','family','num_channels','latent_k','seed','length','horizon',
                'requirement','execution_status','training_status','validity_status','scientific_status','publication_status',
                'reason','planned_epoch_endpoints','resume_command']
        with (self.root/'EXPERIMENT_REGISTRY.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for identity in (self.parent,self.identity):
                writer.writerow(dict.fromkeys(fields,'')|dict(experiment_id=identity,dataset='beijing',fold='primary',block='E4',
                    family='ORDERED_MULTI_K',num_channels='11',latent_k='11',seed='92601',length='512',horizon='48'))

    def put(self,relative,text):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')

    def test_ordered_v_detected_and_unknown_cache_flag_preserved(self):
        bundle=self.generator.capture(self.root,[self.identity])
        row=self.generator.load_tables(bundle)['COMPARISONS.csv'][0]
        self.assertTrue(row['new_V_present'])
        self.assertEqual(row['V_status'],'V_METADATA_AVAILABLE')
        self.assertIsNone(row['new_V_cache_reused_recorded'])
        self.assertEqual(row['family'],'B0')
        self.assertEqual(row['label'],'Shared ordered K')

    def test_duplicate_canonical_candidates_are_rejected(self):
        manifest=copy.deepcopy(self.manifests[self.identity]);meta=manifest['metadata']
        meta['artifact_sha256']=meta.pop('parent_artifact_sha256');meta['reused_prediction_cache']=False
        self.put('predictions/'+self.identity+'/V_select/PREDICTION_MANIFEST.json',json.dumps(manifest))
        with self.assertRaises(ValueError):self.generator.capture(self.root,[self.identity])

    def test_wrong_run_artifact_split_or_cohort_rejected(self):
        for field in ('run_id','artifact','split','cohort'):
            with self.subTest(field=field):
                manifest=copy.deepcopy(self.manifests[self.identity])
                if field=='artifact':manifest['metadata']['parent_artifact_sha256']='wrong'
                else:manifest[field]='wrong'
                self.put('ordered_predictions/'+self.identity+'/PREDICTION_MANIFEST.json',json.dumps(manifest))
                with self.assertRaises(ValueError):self.generator.capture(self.root,[self.identity])

    def test_originally_missing_stays_missing_when_live_file_appears_later(self):
        path=self.root/'ordered_predictions'/self.identity/'PREDICTION_MANIFEST.json';path.unlink()
        bundle=self.generator.capture(self.root,[self.identity])
        path.write_text(json.dumps(self.manifests[self.identity]),encoding='utf-8')
        row=self.generator.load_tables(bundle)['COMPARISONS.csv'][0]
        self.assertFalse(row['new_V_present']);self.assertEqual(row['V_status'],'TRAIN_ONLY')

    def test_forged_sealed_map_is_rejected(self):
        bundle=self.generator.capture(self.root,[self.identity])
        bundle['prediction_manifest_paths'][self.identity]['relative_path']='ordered_predictions/'+self.parent+'/PREDICTION_MANIFEST.json'
        with self.assertRaises(ValueError):self.generator.load_tables(bundle)

    def test_registry_display_family_must_match_source_ordered_setting(self):
        source=self.root/'runs'/self.identity/'SOURCE_CONTRACT.json'
        value=json.loads(source.read_text());value['spec']['k_grid']=[11];source.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.generator.capture(self.root,[self.identity])

    def test_legacy_v2_sealed_tables_are_byte_identical(self):
        old=module('generate_v2.py')
        bundle=json.loads((HERE/'snapshot_1792082073095310300/INPUT_BUNDLE.json').read_text(encoding='utf-8'))
        old_tables=old.load_tables(bundle);new_tables=self.generator.load_tables(bundle)
        self.assertEqual(set(old_tables),set(new_tables))
        for name in old_tables:
            self.assertEqual(old.table_bytes(old_tables[name]),self.generator.table_bytes(new_tables[name]),name)

if __name__=='__main__':unittest.main()
