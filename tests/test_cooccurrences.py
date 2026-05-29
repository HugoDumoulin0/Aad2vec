import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cooccurrences import (
    characteristic_contexts_for_word_cooc,
    top_cooccurring_words,
    top_similar_words_by_cooc,
)


class CooccurrenceTests(unittest.TestCase):
    def test_top_cooccurring_words_ranks_by_ppmi_and_keeps_raw_counts(self):
        sentences = [
            ["travail", "rare"],
            ["travail", "commun"],
            ["travail", "commun"],
            ["travail", "commun"],
            ["autre", "commun"],
            ["autre", "commun"],
            ["autre", "commun"],
            ["autre", "commun"],
            ["autre", "commun"],
        ]

        df = top_cooccurring_words(
            sentences=sentences,
            target_word="travail",
            window_size=1,
            topn=3,
        )

        self.assertEqual(df.iloc[0]["word"], "rare")
        self.assertGreater(float(df.iloc[0]["ppmi"]), float(df.iloc[1]["ppmi"]))
        self.assertEqual(int(df[df["word"] == "commun"].iloc[0]["cooccurrences"]), 3)
        self.assertIn("ppmi", df.columns)
        self.assertNotIn("travail", df["word"].tolist())

    def test_characteristic_contexts_are_scored_by_matched_ppmi(self):
        sentences = [
            ["travail", "rare"],
            ["travail", "commun"],
            ["travail", "commun", "rare"],
            ["autre", "commun"],
            ["autre", "commun"],
            ["autre", "commun"],
        ]

        df = characteristic_contexts_for_word_cooc(
            sentences=sentences,
            target_word="travail",
            window_size=1,
            topn_words=3,
            topn_contexts=2,
        )

        self.assertEqual(df.iloc[0]["context"], "travail commun rare")
        self.assertEqual(int(df.iloc[0]["n_matched"]), 2)
        self.assertIn("score", df.columns)

    def test_characteristic_contexts_display_surface_forms(self):
        sentences = [
            {
                "tokens": ["travail", "commun", "rare"],
                "word": ["Travaux", "communs", "rares"],
            },
            {
                "tokens": ["autre", "commun"],
                "word": ["autres", "communs"],
            },
        ]

        df = characteristic_contexts_for_word_cooc(
            sentences=sentences,
            target_word="travail",
            window_size=1,
            topn_words=3,
            topn_contexts=1,
        )

        self.assertEqual(df.iloc[0]["context"], "Travaux communs rares")

    def test_top_similar_words_by_cooc_uses_ppmi_row_similarity(self):
        sentences = [
            ["ouvrier", "travail", "salaire"],
            ["ouvrier", "emploi", "salaire"],
            ["chat", "loisir", "chien"],
        ]

        df = top_similar_words_by_cooc(
            sentences=sentences,
            target_word="travail",
            window_size=1,
            topn=2,
        )

        self.assertEqual(df.iloc[0]["word"], "emploi")
        self.assertGreater(float(df.iloc[0]["similarity"]), 0.0)


if __name__ == "__main__":
    unittest.main()
