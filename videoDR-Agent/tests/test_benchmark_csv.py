"""CSV regression tests without importing API clients or loading credentials."""
import ast
import tempfile
import unittest
from pathlib import Path

import pandas as pd

source = Path(__file__).resolve().parents[1] / "eval.py"
function = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "load_benchmark_csv")
namespace = {"pd": pd}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
load_benchmark_csv = namespace["load_benchmark_csv"]


class BenchmarkCsvTest(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text(text, encoding="utf-8")
            return load_benchmark_csv(path)

    def test_current_schema_selects_named_columns(self):
        result = self.load('id,category,video_url,question,answer,evidence,remarks,difficulty\n'
                           '1,History,https://example.org/video,"Who, when?",0012,evidence,,High\n')
        self.assertEqual(result.to_dict("records"),
                         [{"index": 1, "question": "Who, when?", "answer": "0012"}])

    def test_reordered_columns_bom_and_multiline(self):
        result = self.load('\ufeffanswer,question,id,category\nNA,"first line\nsecond line",500,History\n')
        self.assertEqual(result.to_dict("records"),
                         [{"index": 500, "question": "first line\nsecond line", "answer": "NA"}])

    def test_legacy_preserves_first_row_and_answer_strings(self):
        result = self.load('1,First question,0012\n2,Second question,NA\n')
        self.assertEqual(result["index"].tolist(), [1, 2])
        self.assertEqual(result["answer"].tolist(), ["0012", "NA"])

    def test_named_index(self):
        self.assertEqual(self.load('index,question,answer\n7,Q,A\n')["index"].tolist(), [7])

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load('id,question,answer\n1,Q,A\n1,Q2,A2\n')

    def test_noninteger_ids_rejected(self):
        for value in ['1.5', 'invalid', '']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(f'id,question,answer\n{value},Q,A\n')


if __name__ == "__main__":
    unittest.main()
