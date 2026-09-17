"""
AR Emoji Filter berbasis ekspresi wajah.

Contoh:
    python scripts/ar_filter.py
    python scripts/ar_filter.py --source 1
    python scripts/ar_filter.py --source video.mp4 --no-smooth
"""

import argparse
import os
import sys

# Add src and root to path (lightweight — tetap di top-level agar --help
# jalan tanpa heavy deps, lihat AGENTS.md)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _lazy_imports():
    """Import berat (cv2/numpy/modul lokal) secara lazy."""
    global cv2, np
    global config, MultiFaceSmoother, EmotionPredictor
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import cv2 as _cv2
    import numpy as _np
    import config as _config
    from smoothing import MultiFaceSmoother as _MFS
    from inference import EmotionPredictor as _EP
    cv2, np = _cv2, _np
    config = _config
    MultiFaceSmoother = _MFS
    EmotionPredictor = _EP


class ARFilter:
    """
    Class untuk membuat filter AR berbasis ekspresi wajah
    """

    def __init__(self):
        _lazy_imports()
        print("Memuat model dan aset filter...")

        # Single inference path (AGENTS.md): prediksi via EmotionPredictor
        # (.h5 eksplisit; FileNotFoundError ditangani di __main__).
        self.predictor = EmotionPredictor(
            model_path=config.MODEL_PATH,
            detection_confidence=config.DETECTION_CONFIDENCE,
        )
        print(f"Model dimuat dari: {self.predictor.model_path}")
        print(f"Face detector: {self.predictor.detector_name}")
        self.smoother = MultiFaceSmoother(window_size=7, ema_alpha=0.4)

        # Load emoji assets
        self.assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'filters')
        self.filters = {}

        # Mapping dari label model ke nama file emoji (dua varian nama didukung)
        mapping = {
            'marah': ['angry.png', 'marah.png'],
            'jijik': ['disgust.png', 'jijik.png'],
            'takut': ['fear.png', 'takut.png'],
            'senang': ['happy.png', 'senang.png'],
            'netral': ['neutral.png', 'netral.png'],
            'sedih': ['sad.png', 'sedih.png'],
            'kaget': ['surprise.png', 'kaget.png'],
        }

        for label, candidates in mapping.items():
            for filename in candidates:
                path = os.path.join(self.assets_dir, filename)
                if os.path.exists(path):
                    # Load with alpha channel (UNCHANGED)
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.filters[label] = img
                        print(f"Filter dimuat: {label} <- {filename}")
                        break
            else:
                print(f"Warning: File filter tidak ditemukan untuk '{label}': {candidates}")

    def overlay_transparent(self, background, overlay, x, y, size):
        """
        Menempelkan gambar transparan (overlay) ke background dengan kualitas tinggi
        """
        if overlay is None or size <= 0:
            return background

        # Gunakan INTER_AREA untuk downscaling agar tidak pecah (pixelated)
        # Atau INTER_CUBIC untuk upscaling
        interp = cv2.INTER_AREA if overlay.shape[0] > size else cv2.INTER_CUBIC
        overlay = cv2.resize(overlay, (size, size), interpolation=interp)

        h, w = overlay.shape[:2]

        # Koordinat clipping agar tidak error jika emoji keluar layar
        x1, x2 = max(0, x), min(background.shape[1], x + w)
        y1, y2 = max(0, y), min(background.shape[0], y + h)

        # Ukuran area yang terlihat
        overlap_w = x2 - x1
        overlap_h = y2 - y1

        if overlap_w <= 0 or overlap_h <= 0:
            return background

        # Bagian overlay yang masuk dalam frame
        overlay_x1 = max(0, -x)
        overlay_y1 = max(0, -y)
        overlay_part = overlay[overlay_y1:overlay_y1 + overlap_h, overlay_x1:overlay_x1 + overlap_w]

        # Ambil alpha channel untuk blending
        if overlay_part.shape[2] < 4:
            background[y1:y2, x1:x2] = overlay_part[:, :, :3]
            return background

        overlay_img = overlay_part[:, :, :3]
        overlay_mask = overlay_part[:, :, 3:] / 255.0

        background_roi = background[y1:y2, x1:x2]

        # Alpha blending
        background[y1:y2, x1:x2] = (1.0 - overlay_mask) * background_roi + overlay_mask * overlay_img

        return background

    def _resolve_source(self, source):
        """Normalisasi --source: None -> config, '0'/'1' -> int, path -> str."""
        if source is None:
            return config.CAMERA_INDEX
        if isinstance(source, int):
            return source
        s = str(source).strip()
        if s.isdigit():
            return int(s)
        return s

    def run(self, source=None, use_smoothing=True):
        video_source = self._resolve_source(source)
        is_camera = isinstance(video_source, int)

        cap = cv2.VideoCapture(video_source)

        # Pengaturan kualitas kamera (HD 720p)
        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Gunakan MJPG untuk performa FPS yang lebih baik di Windows
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka sumber video: {video_source!r}")
            if is_camera:
                print("Tips: coba --source 1 atau cek config.CAMERA_INDEX di src/config.py")
            return

        print("\n" + "=" * 50)
        print("AR EMOJI FILTER")
        print(f"Sumber: {'Webcam ' + str(video_source) if is_camera else video_source}")
        print(f"Smoothing: {'ON' if use_smoothing else 'OFF'}")
        print("=" * 50)
        print("Tekan 'q' untuk keluar")

        while True:
            ret, frame = cap.read()
            if not ret:
                # Jika video file habis, loop kembali ke awal
                if not is_camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame = cv2.flip(frame, 1)
            # Deteksi + prediksi via shared backend (terbesar dulu agar
            # slot smoother stabil; margin 0.0 seperti crop lama tanpa margin)
            try:
                results = self.predictor.predict_faces(frame, margin=0.0)
            except Exception as e:
                print(f"Warning prediksi gagal: {e}")
                results = []

            if use_smoothing and results:
                raw_preds = [(r["label"], r["confidence"],
                              np.array([r["probs"][lb]
                                        for lb in self.predictor.ordered_labels]),
                              self.predictor.ordered_labels) for r in results]
                smooth = self.smoother.update(raw_preds)
                merged = [{**r, "label": lb, "confidence": cf}
                          for r, (lb, cf) in zip(results, smooth)]
            else:
                if not use_smoothing:
                    self.smoother.reset()
                merged = results

            for r in merged:
                x, y, w, h = r["box"]
                label = r["label"]
                # Ambil emoji yang sesuai
                emoji = self.filters.get(label)

                if emoji is not None:
                    # Tentukan posisi: di atas kepala
                    emoji_size = int(w * 0.8)
                    pos_x = x + (w - emoji_size) // 2
                    pos_y = max(0, y - emoji_size - 10)

                    frame = self.overlay_transparent(frame, emoji, pos_x, pos_y, emoji_size)

                # Tambahkan teks label di bawah emoji (opsional)
                cv2.putText(frame, label.upper(), (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Emoji Filter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="AR Emoji Filter berbasis ekspresi")
    parser.add_argument("--source", help="Index webcam, path video, atau URL stream (default: config.CAMERA_INDEX)", default=None)
    parser.add_argument("--no-smooth", action="store_true", help="Matikan temporal smoothing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        filter_app = ARFilter()
        filter_app.run(source=args.source, use_smoothing=not args.no_smooth)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
