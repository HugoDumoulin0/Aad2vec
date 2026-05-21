import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pca_w2v import add_pca_contributions, filter_top_contributors, make_pos_tag


class PcaW2VTests(unittest.TestCase):
    def test_make_pos_tag_is_stable_for_empty_and_sorted_values(self):
        self.assertEqual(make_pos_tag(None), "ALL")
        self.assertEqual(make_pos_tag(["VERB", "NOUN"]), "NOUN-VERB")

    def test_add_pca_contributions_adds_weighted_total(self):
        df = pd.DataFrame(
            {
                "word": ["a", "b"],
                "x": [2.0, 0.0],
                "y": [0.0, 2.0],
            }
        )

        out = add_pca_contributions(
            df,
            {"explained_variance_ratio": [0.75, 0.25]},
        )

        self.assertIn("contrib_dim1", out.columns)
        self.assertIn("contrib_dim2", out.columns)
        self.assertIn("contrib_total", out.columns)
        self.assertAlmostEqual(out.loc[0, "contrib_total"], 0.75)
        self.assertAlmostEqual(out.loc[1, "contrib_total"], 0.25)

    def test_filter_top_contributors_keeps_highest_rows(self):
        df = pd.DataFrame(
            {
                "word": ["low", "mid", "high", "top"],
                "contrib_total": [0.1, 0.2, 0.3, 0.4],
            }
        )

        out = filter_top_contributors(df, pct=50)

        self.assertEqual(out["word"].tolist(), ["top", "high"])


if __name__ == "__main__":
    unittest.main()
