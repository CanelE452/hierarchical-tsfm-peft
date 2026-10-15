"""Style-only v2: immutable v1 and all non-figure code remain identical."""
from pathlib import Path
import ast
import hashlib
import unittest


HERE=Path(__file__).resolve().parent
OLD_Y='ax.set_ylabel("Complete-C FP32 MSE minus numeric minimum")'
NEW_Y='ax.set_ylabel("Normalized MSE gap")'
OLD_TITLE='Same D and mask for each cell; FP32/real trace false inequalities and discarded components are preserved separately'
NEW_TITLE='Complete-C FP32 MSE minus retained numeric minimum\\nSame D/mask per cell; real-trace/FP32 false inequalities and discarded components preserved separately'


class StyleTests(unittest.TestCase):
    def sources(self):
        before=(HERE/"make_learned_output_space_report_v1.py").read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),"3076834bd0bb50eb3b161baf11a493de5e8b7838625b7a0c861942405d2946ab")
        path=HERE/"make_learned_output_space_report_v2.py"
        self.assertTrue(path.exists(),"Fresh style-only v2 source is missing")
        return before.decode("utf-8"),path.read_text(encoding="utf-8")
    def test_exact_two_visual_strings_only(self):
        before,after=self.sources()
        self.assertEqual(before.count(OLD_Y),1); self.assertEqual(before.count(OLD_TITLE),1)
        self.assertEqual(after,before.replace(OLD_Y,NEW_Y).replace(OLD_TITLE,NEW_TITLE))
        compile(after,"style_only_v2","exec")
    def test_all_nonfigure_AST_identical(self):
        before,after=self.sources(); trees=[ast.parse(s) for s in (before,after)]
        for tree in trees: tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=="figures")]
        self.assertEqual(ast.dump(trees[0]),ast.dump(trees[1]))


if __name__=="__main__": unittest.main()
