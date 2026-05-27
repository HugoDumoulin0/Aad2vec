import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import loaders


class LoaderTests(unittest.TestCase):
    def test_load_sentences_supports_legacy_list_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentences_dir = Path(tmp) / "sentences"
            sentences_dir.mkdir()
            (sentences_dir / "gauche_sentences.json").write_text(
                json.dumps([["le", "travail"], ["la", "republique"]]),
                encoding="utf-8",
            )

            with patch.object(loaders, "run_dirs", return_value={"sentences": str(sentences_dir)}):
                sentences = loaders.load_sentences("run-test", "gauche")

        self.assertEqual(sentences[0]["tokens"], ["le", "travail"])
        self.assertEqual(sentences[0]["word"], ["le", "travail"])
        self.assertEqual(sentences[0]["surface_text"], "le travail")

    def test_load_sentences_supports_structured_format(self):
        payload = [
            {
                "tokens": ["nation"],
                "word": ["Nation"],
                "lemma": ["nation"],
                "surface_text": "Nation",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            sentences_dir = Path(tmp) / "sentences"
            sentences_dir.mkdir()
            (sentences_dir / "centre_sentences.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with patch.object(loaders, "run_dirs", return_value={"sentences": str(sentences_dir)}):
                sentences = loaders.load_sentences("run-test", "centre")

        self.assertEqual(sentences, payload)


if __name__ == "__main__":
    unittest.main()
