import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import normalize_rows, sentence_tokens, stable_short_hash


class UtilsTests(unittest.TestCase):
    def test_sentence_tokens_accepts_legacy_and_structured_sentences(self):
        self.assertEqual(sentence_tokens(["un", "deux"]), ["un", "deux"])
        self.assertEqual(
            sentence_tokens({"tokens": ["trois", "quatre"], "word": ["Trois", "quatre"]}),
            ["trois", "quatre"],
        )

    def test_normalize_rows_keeps_zero_rows_stable(self):
        matrix = np.array([[3.0, 4.0], [0.0, 0.0]])

        normalized = normalize_rows(matrix)

        np.testing.assert_allclose(normalized[0], np.array([0.6, 0.8]))
        np.testing.assert_allclose(normalized[1], np.array([0.0, 0.0]))

    def test_stable_short_hash_is_deterministic_and_short(self):
        first = stable_short_hash("run|GLOBAL|NOUN|100")
        second = stable_short_hash("run|GLOBAL|NOUN|100")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)


if __name__ == "__main__":
    unittest.main()
