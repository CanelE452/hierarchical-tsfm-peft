import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('multik_lossless', HERE / 'bundle_multik.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bundle.trial_bytes(value))


class LosslessTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'costs' / 'synthetic_cost'
        folder = source / 'synthetic_B1_shared' / 'shared'
        rows = [dict(phase='warmup', repeat=0, from_K=1, to_K=2,
                     elapsed=0.12345678901234567, note='synthetic',
                     nested={'guard': None}, resource_within_limit=True),
                dict(phase='timed', repeat=0, from_K=2, to_K=1,
                     elapsed=1.2345678901234567e-8, note='synthetic',
                     nested={'guard': None}, resource_within_limit=True)]
        ledger = [dict(phase=row['phase'], from_K=row['from_K'], to_K=row['to_K'],
                       requested=1, actual_complete=1, remaining=0,
                       completed_within_resource=1, remaining_within_resource=0)
                  for row in rows]
        result = dict(rows=rows, transition_ledger=ledger, exact_minimum_protocol_complete=True,
                      timer_scope='instrumented synthetic', memory={'rss': 123},
                      source={'synthetic.py': 'not_an_actual_scientific_source'})
        write_json(folder / 'RESULT.json', result)
        write_json(folder / 'NUMERIC_EXECUTION.json', dict(result, final_protocol_decision=False))
        import csv
        with (folder / 'TRIALS.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        for index, row in enumerate(rows):
            write_json(folder / 'actual_trials' / f'trial_{index:06}.json', row)
        write_json(source / 'FULL_QUEUE_PLAN.json', {'scope_count': 20, 'synthetic': True})
        write_json(source / 'BATCH_LAUNCHER_RECEIPT.json', {'actual': 1, 'remaining': 19})
        return source

    def build(self, root):
        source = self.fixture(root)
        output = root / 'package'
        manifest = bundle.build(root, source, output, expected_scopes=1, expected_trials=2,
                                writer_pin={'sha256': 'synthetic_source_only'})
        return source, output, manifest

    def test_exact_roundtrip_repeat_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source, output, manifest = self.build(root)
            before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            target = root / 'clone'
            report = bundle.restore(output / 'MANIFEST.json', target)
            self.assertEqual(report['restored'], len(before))
            self.assertEqual(report['trial_sha_matches'], 2)
            self.assertEqual(bundle.restore(output / 'MANIFEST.json', target)['already_matching'], len(before))
            for rel, raw in before.items():
                self.assertEqual((target / manifest['source_relative'] / rel).read_bytes(), raw)
                self.assertEqual((source / rel).read_bytes(), raw)

    def test_wrong_existing_preflight_writes_nothing_else(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source, output, manifest = self.build(root)
            target = root / 'clone'; bad = target / manifest['source_relative'] / 'FULL_QUEUE_PLAN.json'
            bad.parent.mkdir(parents=True); bad.write_bytes(b'wrong existing file')
            with self.assertRaisesRegex(ValueError, 'existing'):
                bundle.restore(output / 'MANIFEST.json', target)
            self.assertEqual([p for p in target.rglob('*') if p.is_file()], [bad])

    def test_tampered_payload_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source, output, manifest = self.build(root)
            record = next(x for x in manifest['records'] if x['kind'] == 'gzip')
            (output / record['payload']).write_bytes(b'corrupt gzip')
            target = root / 'clone'
            with self.assertRaisesRegex(ValueError, 'payload'):
                bundle.restore(output / 'MANIFEST.json', target)
            self.assertFalse(target.exists())

    def test_wrong_trial_order_or_precision_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = self.fixture(root)
            path = next(source.rglob('trial_000000.json'))
            row = json.loads(path.read_text(encoding='utf-8'))
            row['elapsed'] = round(row['elapsed'], 4)
            path.write_bytes(bundle.trial_bytes(row))
            with self.assertRaisesRegex(ValueError, 'trial'):
                bundle.build(root, source, root / 'package', 1, 2, {})
            self.assertFalse((root / 'package').exists())

    def test_wrong_crlf_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = self.fixture(root)
            path = next(source.rglob('trial_000000.json'))
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
            with self.assertRaisesRegex(ValueError, 'byte'):
                bundle.build(root, source, root / 'package', 1, 2, {})

    def test_csv_or_ledger_mismatch_rejected(self):
        for kind in ['csv', 'ledger']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as name:
                root = Path(name); source = self.fixture(root)
                if kind == 'csv':
                    path = next(source.rglob('TRIALS.csv'))
                    path.write_bytes(path.read_bytes().replace(b'0.12345678901234566', b'0.5'))
                else:
                    path = next(source.rglob('RESULT.json')); value = json.loads(path.read_text())
                    value['transition_ledger'][0]['actual_complete'] = 0
                    path.write_bytes(bundle.trial_bytes(value))
                with self.assertRaisesRegex(ValueError, kind):
                    bundle.build(root, source, root / 'package', 1, 2, {})

    def test_unsafe_manifest_target_and_count_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source, output, manifest = self.build(root)
            manifest['records'][0]['relative'] = '../escape.json'
            (output / 'MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'relative'):
                bundle.restore(output / 'MANIFEST.json', root / 'clone')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'count'):
                bundle.build(root, source, root / 'package', 8, 1440, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
