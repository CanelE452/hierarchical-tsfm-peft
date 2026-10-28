import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.ppt_justification_20261028_v1.package import archived_public_files, cost_public_files, e_public_files


class SupplementBundleTests(unittest.TestCase):
    def test_only_explicit_e_sources_public_reports_and_registered_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp)
            allowed=['research/ppt_e_fixed_20261028_v1/models.py',
                'research/ppt_e_fixed_20261028_v1/tests/test_postprocess.py',
                'reports/ppt_e_fixed_20261028_v1/PREREGISTRATION.json',
                'reports/ppt_e_fixed_20261028_v1/AMENDMENT_001.json',
                'reports/ppt_e_fixed_20261028_v1/FIG_E_FIXED.png',
                'reports/ppt_e_fixed_20261028_v1/ALL_NEW_RUNS.csv',
                'reports/ppt_e_fixed_20261028_v1/jobs/PE_E_FIXED_jena_K6_s92601/RESULT.json',
                'reports/ppt_e_fixed_20261028_v1/jobs/PE_E_FIXED_jena_K6_s92601/attempt_001/SELECTION.json',
                'reports/ppt_e_fixed_20261028_v1/jobs/PE_E_FIXED_jena_K6_s92601/attempt_001/HOST_GPU_GATE.json',
                'reports/ppt_e_fixed_20261028_v1/jobs/PE_E_FIXED_jena_K6_s92601/attempt_001/E_INVARIANCE.json']
            denied=['research/ppt_e_fixed_20261028_v1/unknown.py',
                'reports/ppt_e_fixed_20261028_v1/unknown.json',
                'reports/ppt_e_fixed_20261028_v1/predictions.npz',
                'reports/ppt_e_fixed_20261028_v1/checkpoint.pt',
                '.cache/ppt_e_fixed_20261028_v1/RESULT.json',
                '.cache/ppt_e_fixed_20261028_v1/host_gpu_policy.json',
                'reports/ppt_e_fixed_20261028_v1/jobs/PE_E_FIXED_jena_K6_s92601/attempt_001/raw_gpu_counter.json',
                'reports/ppt_e_fixed_20261028_v1/jobs/PE_E_FIXED_jena_K6_s92601/attempt_003/HOST_GPU_GATE.json',
                'reports/ppt_e_fixed_20261028_v1/jobs/OTHER_JOB/RESULT.json']
            for relative in allowed+denied:
                p=repo/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic test fixture',encoding='utf8')
            actual={p.relative_to(repo).as_posix() for p in e_public_files(repo,{'appendix':True})}
            self.assertEqual(actual,set(allowed))

    def test_exclusion_does_not_bundle_an_unverified_score_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp);report=repo/'reports/ppt_e_fixed_20261028_v1';report.mkdir(parents=True)
            for name in ('FIG_E_FIXED.png','SUMMARY.csv','RESULTS_KO.md','BUDGET_DECISION.json'):
                (report/name).write_text('synthetic test fixture',encoding='utf8')
            disposition={'appendix':False,'provenance':{'status_evidence_path':'reports/ppt_e_fixed_20261028_v1/BUDGET_DECISION.json'}}
            actual={p.name for p in e_public_files(repo,disposition)}
            self.assertEqual(actual,{'RESULTS_KO.md','BUDGET_DECISION.json'})

    def test_exclusion_evidence_cannot_escape_its_campaign(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                e_public_files(Path(temp),{'provenance':{'status_evidence_path':'reports/other/private.json'}})

    def test_cost_scope_is_explicit_and_excludes_drafts_and_private_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp);prefix='reports/ppt_justification_20261028_v1/cost_profile/'
            names=['COST_PROTOCOL.json','COST_SEAL.json','COST_INPUTS.json','CPU_TESTS.json',
                'AMENDMENT_001.json','AMENDMENT_002.json','HOST_GPU_CONTRACT.json','HOST_GATE_CPU_TESTS.json',
                'HOST_CODE_REVIEW.json','ROOT_LAUNCH_GATE.json','COST_LAUNCH.json','COST_SUMMARY.json','COST_VALIDATION.json']
            names += [facet+'_'+suffix+'.json' for facet in
                ('Raw_appliances','KTAU_main_appliances_K10_s92601','KTAU_main_appliances_K10_s92602')
                for suffix in ('MEASUREMENT','HOST_PREFLIGHT','HOST_OBSERVATION')]
            allowed=[prefix+n for n in names]
            denied=[prefix+n for n in ('COST_PROTOCOL_v3_DRAFT.json','policy.json','identity.json','raw_counter.json',
                'stderr.log','predictions.npz','checkpoint.pt','unknown_HOST_PREFLIGHT.json',
                'nested/COST_SUMMARY.json','.cache/HOST_GPU_CONTRACT.json')]
            denied+=['.cache/ppt_justification_20261028_v1/cost_profile/Raw_appliances_HOST_OBSERVATION.json']
            for relative in allowed+denied:
                p=repo/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture',encoding='utf8')
            actual={p.relative_to(repo).as_posix() for p in cost_public_files(repo)}
            self.assertEqual(actual,set(allowed))

    def make_archive(self,repo,archive,source,nested=True):
        relative=archive+'/'+(source if nested else Path(source).name)
        p=repo/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'preserved original\n')
        row={'source':source,'archived':relative,'bytes':p.stat().st_size,
             'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        receipt=repo/archive/'ARCHIVE_RECEIPT.json'
        receipt.write_text(json.dumps({'files':[row]}),encoding='utf8')
        return p,receipt,row

    def test_archive_includes_only_verified_public_python_and_receipt(self):
        for nested in (False,True):
            with self.subTest(nested=nested),tempfile.TemporaryDirectory() as temp:
                repo=Path(temp);archive='reports/cost/revisions/preflight_001';source='research/cost/profile.py'
                p,receipt,row=self.make_archive(repo,archive,source,nested)
                (repo/archive/'unlisted.json').write_text('private',encoding='utf8')
                self.assertEqual(archived_public_files(repo,archive,{source}),{p,receipt})

    def test_archive_rejects_missing_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp);archive='reports/cost/revisions/preflight_001'
            (repo/archive).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'ARCHIVE_RECEIPT'):
                archived_public_files(repo,archive,{'research/cost/profile.py'})

    def test_archive_rejects_unbound_or_private_members(self):
        changes=[{'archived':'../private.json'},{'archived':'C:/private.json'},
                 {'archived':'reports/cost/revisions/preflight_002/other.json'},
                 {'source':'.cache/policy.json'},{'source':'research/cost/predictions.npz'},
                 {'bytes':999},{'sha256':'0'*64}]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                repo=Path(temp);archive='reports/cost/revisions/preflight_001';source='research/cost/profile.py'
                p,receipt,row=self.make_archive(repo,archive,source)
                row.update(change);receipt.write_text(json.dumps({'files':[row]}),encoding='utf8')
                with self.assertRaises(ValueError):archived_public_files(repo,archive,{source})

    def test_archive_rejects_unsupported_suffix_even_if_named_in_allowlist(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp);archive='reports/cost/revisions/preflight_001';source='research/cost/predictions.npz'
            self.make_archive(repo,archive,source)
            with self.assertRaisesRegex(ValueError,'suffix'):archived_public_files(repo,archive,{source})

    def test_archive_rejects_duplicate_missing_or_modified_originals(self):
        for condition in ('duplicate','missing','modified'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as temp:
                repo=Path(temp);archive='reports/cost/revisions/preflight_001';source='research/cost/profile.py'
                p,receipt,row=self.make_archive(repo,archive,source)
                if condition=='duplicate':receipt.write_text(json.dumps({'files':[row,row]}),encoding='utf8')
                elif condition=='missing':p.unlink()
                else:p.write_bytes(b'changed original!!!')
                with self.assertRaises(ValueError):archived_public_files(repo,archive,{source})

    def test_archive_collection_covers_cost_both_revisions_and_e_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp);cost='reports/ppt_justification_20261028_v1/cost_profile'
            first=self.make_archive(repo,cost+'/revisions/preflight_001','research/ppt_justification_20261028_v1/cost_profile.py',False)
            second=self.make_archive(repo,cost+'/revisions/preflight_002','research/ppt_justification_20261028_v1/verify_cost_results.py')
            e=self.make_archive(repo,'reports/ppt_e_fixed_20261028_v1/revisions/preflight_001','research/ppt_e_fixed_20261028_v1/run_one.py')
            self.assertEqual(cost_public_files(repo),{first[0],first[1],second[0],second[1]})
            self.assertEqual(e_public_files(repo,{'appendix':False}),{e[0],e[1]})


if __name__=='__main__':unittest.main()
