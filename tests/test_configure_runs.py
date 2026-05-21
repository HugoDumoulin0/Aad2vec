import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from configure_runs import build_groups_from_metadata, detect_file_column, normalize_filename


class ConfigureRunsTests(unittest.TestCase):
    def test_normalize_filename_removes_extension_and_path(self):
        self.assertEqual(
            normalize_filename("/tmp/EL009_L_1958_11_007_01_1_PF_01.txt"),
            "el009_l_1958_11_007_01_1_pf_01",
        )

    def test_detect_file_column_prefers_known_names(self):
        df = pd.DataFrame(
            {
                "id": ["doc_1"],
                "orientation": ["gauche"],
            }
        )

        self.assertEqual(detect_file_column(df, ["doc_1.txt"]), "id")

    def test_build_groups_from_metadata_matches_files_without_extension(self):
        df = pd.DataFrame(
            {
                "id": ["doc_1", "doc_2", "missing"],
                "orientation": ["gauche", "droite", "centre"],
            }
        )

        groups = build_groups_from_metadata(
            df=df,
            file_col="id",
            segment_col="orientation",
            corpus_files=["doc_1.txt", "doc_2.txt"],
        )

        self.assertEqual(groups, {"droite": ["doc_2.txt"], "gauche": ["doc_1.txt"]})


if __name__ == "__main__":
    unittest.main()
