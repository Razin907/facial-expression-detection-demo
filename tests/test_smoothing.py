"""Unit test untuk src/smoothing.py (stdlib only, tanpa TF/cv2).

Jalankan dari repo root:
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smoothing import MultiFaceSmoother, PredictionSmoother

CLASSES = ["marah", "jijik", "takut", "senang", "netral", "sedih", "kaget"]


def probs_for(idx, n=7, high=0.9):
    low = (1.0 - high) / (n - 1)
    return [high if i == idx else low for i in range(n)]


class TestPredictionSmoother(unittest.TestCase):
    def test_majority_vote(self):
        s = PredictionSmoother(window_size=3, ema_alpha=0.4)
        s.update("senang")
        s.update("senang")
        label, conf = s.update("marah")
        self.assertEqual(label, "senang")
        self.assertAlmostEqual(conf, 2 / 3)

    def test_window_slides(self):
        s = PredictionSmoother(window_size=3, ema_alpha=0.4)
        for lb in ["marah", "marah", "marah", "senang", "senang", "senang"]:
            label, _ = s.update(lb)
        self.assertEqual(label, "senang")

    def test_window_size_floor(self):
        s = PredictionSmoother(window_size=0, ema_alpha=0.4)
        self.assertEqual(s.window_size, 1)
        label, conf = s.update("netral")
        self.assertEqual((label, conf), ("netral", 1.0))

    def test_ema_confidence(self):
        s = PredictionSmoother(window_size=7, ema_alpha=0.5)
        label, conf = s.update("senang", probs=probs_for(3), classes=CLASSES)
        self.assertEqual(label, "senang")
        self.assertAlmostEqual(conf, 0.9)  # EMA pertama = observasi
        # Update kedua dengan distribusi berbeda: EMA blend 0.5/0.5
        label, conf = s.update("senang", probs=probs_for(4), classes=CLASSES)
        low = (1.0 - 0.9) / 6
        self.assertAlmostEqual(conf, 0.5 * low + 0.5 * 0.9)

    def test_reset(self):
        s = PredictionSmoother(window_size=3, ema_alpha=0.4)
        s.update("marah", probs=probs_for(0), classes=CLASSES)
        s.reset()
        label, conf = s.update("senang")
        self.assertEqual((label, conf), ("senang", 1.0))


class TestMultiFaceSmoother(unittest.TestCase):
    def _raw(self, idx, conf=0.9):
        return (CLASSES[idx], conf, probs_for(idx), CLASSES)

    def test_slots_independent(self):
        m = MultiFaceSmoother(window_size=3, ema_alpha=0.4)
        out = m.update([self._raw(0), self._raw(3)])
        self.assertEqual([lb for lb, _ in out], ["marah", "senang"])
        self.assertEqual(len(m.smoothers), 2)

    def test_shrink_on_fewer_faces(self):
        m = MultiFaceSmoother(window_size=3, ema_alpha=0.4)
        m.update([self._raw(0), self._raw(3)])
        out = m.update([self._raw(4)])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(m.smoothers), 1)

    def test_reset_clears_slots(self):
        m = MultiFaceSmoother(window_size=3, ema_alpha=0.4)
        m.update([self._raw(0)])
        m.reset()
        self.assertEqual(m.smoothers, [])
        out = m.update([self._raw(3)])
        self.assertEqual(out[0][0], "senang")

    def test_no_probs_blends_confidence(self):
        m = MultiFaceSmoother(window_size=7, ema_alpha=0.4)
        out = m.update([("marah", 0.8, None, CLASSES)])
        # vote conf 1.0 dicampur confidence mentah: 0.5*1.0 + 0.5*0.8
        self.assertEqual(out[0][0], "marah")
        self.assertAlmostEqual(out[0][1], 0.9)


if __name__ == "__main__":
    unittest.main()
