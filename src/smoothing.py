"""
Temporal smoothing untuk prediksi ekspresi real-time.

Tanpa smoothing, label berkedip setiap frame karena noise model.
Dua strategi yang disediakan:
- Majority vote dalam sliding window (stabil untuk label)
- EMA (exponential moving average) untuk probabilitas (stabil untuk confidence/bar)

Untuk multi-face, gunakan `MultiFaceSmoother` yang memelihara satu
smoother per slot wajah (diurutkan berdasarkan ukuran box, terbesar dulu).
Ini aproksimasi yang cukup untuk demo real-time tanpa tracker penuh.
"""

from collections import Counter, deque


class PredictionSmoother:
    """Smoother untuk satu aliran wajah."""

    def __init__(self, window_size=7, ema_alpha=0.4):
        self.window_size = max(1, int(window_size))
        self.ema_alpha = float(ema_alpha)
        self.labels = deque(maxlen=self.window_size)
        self.ema_probs = None
        self.classes = []

    def update(self, label, probs=None, classes=None):
        """Masukkan prediksi mentah, kembalikan (label_halus, conf_halus)."""
        self.labels.append(label)

        if probs is not None and classes is not None:
            self.classes = list(classes)
            probs = list(probs)
            if self.ema_probs is None or len(self.ema_probs) != len(probs):
                self.ema_probs = list(probs)
            else:
                a = self.ema_alpha
                self.ema_probs = [
                    a * p + (1 - a) * e for p, e in zip(probs, self.ema_probs)
                ]

        # Majority vote untuk label
        counts = Counter(self.labels)
        smooth_label = counts.most_common(1)[0][0]

        # Confidence dari EMA jika tersedia, fallback ke frekuensi vote
        if self.ema_probs and self.classes:
            try:
                idx = self.classes.index(smooth_label)
                smooth_conf = float(self.ema_probs[idx])
            except ValueError:
                smooth_conf = counts[smooth_label] / len(self.labels)
        else:
            smooth_conf = counts[smooth_label] / len(self.labels)

        return smooth_label, smooth_conf

    def reset(self):
        self.labels.clear()
        self.ema_probs = None


class MultiFaceSmoother:
    """Bank smoother untuk N wajah sekaligus (per slot index)."""

    def __init__(self, window_size=7, ema_alpha=0.4):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.smoothers = []

    def _ensure(self, n):
        while len(self.smoothers) < n:
            self.smoothers.append(
                PredictionSmoother(
                    window_size=self.window_size, ema_alpha=self.ema_alpha
                )
            )
        if len(self.smoothers) > n:
            self.smoothers = self.smoothers[:n]

    def reset(self):
        for s in self.smoothers:
            s.reset()
        self.smoothers = []

    def update(self, raw_preds):
        """raw_preds: list of (label, conf, probs, classes).

        Returns: list of (smooth_label, smooth_conf).
        """
        self._ensure(len(raw_preds))
        out = []
        for smoother, (label, conf, probs, classes) in zip(
            self.smoothers, raw_preds
        ):
            if probs is None:
                # Tanpa distribusi probabilitas: vote saja
                s_label, s_conf = smoother.update(label)
                # campur dengan confidence mentah agar tidak kaku
                s_conf = 0.5 * s_conf + 0.5 * float(conf)
            else:
                s_label, s_conf = smoother.update(
                    label, probs=probs, classes=classes
                )
            out.append((s_label, s_conf))
        return out
