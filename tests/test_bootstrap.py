import hashlib
import json
from pathlib import Path
import unittest


class SealedReferencesTest(unittest.TestCase):
    def test_zip_documents_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "docs/design_history/MANIFEST.json").read_text())
        for item in manifest["files"]:
            content = (root / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
            self.assertEqual(len(content), item["bytes"])
