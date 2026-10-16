import hashlib
from pathlib import Path
import tempfile
import unittest

import verify_public_manifest as audit


class PublishedManifestTests(unittest.TestCase):
    def test_exact_bytes_and_manifest_size_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'number=0.12345678901234567\r\n'
            (root / 'score.txt').write_bytes(raw)
            manifest = root / 'PUBLIC_MANIFEST.sha256'
            manifest.write_text(hashlib.sha256(raw).hexdigest() + '  score.txt\n', encoding='utf-8')
            result = audit.verify(root, manifest)
            self.assertEqual(result['selected_files_including_manifest'], 2)
            self.assertEqual(result['selected_bytes_including_manifest'], len(raw) + manifest.stat().st_size)

    def test_one_changed_byte_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'score.txt').write_bytes(b'2')
            manifest = root / 'PUBLIC_MANIFEST.sha256'
            manifest.write_text(hashlib.sha256(b'1').hexdigest() + '  score.txt\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                audit.verify(root, manifest)

    def test_escape_and_self_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'PUBLIC_MANIFEST.sha256'
            for name, error in [('../outside.txt', 'Unsafe'), ('PUBLIC_MANIFEST.sha256', 'exclude itself')]:
                with self.subTest(name=name):
                    manifest.write_text('0' * 64 + '  ' + name + '\n', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, error):
                        audit.verify(root, manifest)

    def test_manifest_symlink_is_rejected_before_resolving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'actual.sha256'
            target.write_text('0' * 64 + '  score.txt\n', encoding='utf-8')
            alias = root / 'PUBLIC_MANIFEST.sha256'
            try:
                alias.symlink_to(target.name)
            except OSError as error:
                self.skipTest('File symlink creation unavailable: ' + type(error).__name__)
            with self.assertRaisesRegex(ValueError, 'Symlink'):
                audit.verify(root, alias)


if __name__ == '__main__':
    unittest.main()
