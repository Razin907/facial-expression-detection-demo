"""Unit test untuk src/config.py::load_labels (stdlib only).

Jalankan dari repo root:
    python3 -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config


class TestLoadLabels(unittest.TestCase):
    def setUp(self):
        self._orig = config.LABELS_PATH
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(setattr, config, "LABELS_PATH", self._orig)

    def _write(self, obj=None, raw=None):
        path = os.path.join(self._tmp.name, "class_labels.json")
        with open(path, "w") as f:
            if raw is not None:
                f.write(raw)
            else:
                json.dump(obj, f)
        config.LABELS_PATH = path

    def test_missing_file_falls_back(self):
        config.LABELS_PATH = os.path.join(self._tmp.name, "tidak_ada.json")
        self.assertEqual(config.load_labels(), dict(config.DEFAULT_LABELS))

    def test_train_format_index_to_label(self):
        # Seperti train.py: {v: k for k, v in class_indices.items()}
        # (urutan alfabetis flow_from_directory, bukan DEFAULT_LABELS)
        self._write({0: "jijik", 1: "kaget", 2: "marah", 3: "netral",
                     4: "sedih", 5: "senang", 6: "takut"})
        got = config.load_labels()
        self.assertEqual(got["0"], "jijik")
        self.assertEqual(got["2"], "marah")
        self.assertNotEqual(got, dict(config.DEFAULT_LABELS))

    def test_legacy_format_label_to_index(self):
        self._write({"marah": 0, "jijik": 1, "takut": 2, "senang": 3,
                     "netral": 4, "sedih": 5, "kaget": 6})
        got = config.load_labels()
        self.assertEqual(got["0"], "marah")
        self.assertEqual(len(got), 7)

    def test_corrupt_file_falls_back(self):
        self._write(raw="{bukan json")
        self.assertEqual(config.load_labels(), dict(config.DEFAULT_LABELS))

    def test_wrong_count_falls_back(self):
        self._write({"0": "marah", "1": "jijik"})
        self.assertEqual(config.load_labels(), dict(config.DEFAULT_LABELS))


if __name__ == "__main__":
    unittest.main()
