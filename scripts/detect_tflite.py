"""
Deteksi ekspresi wajah real-time dengan TensorFlow Lite (lebih cepat!)

Contoh:
    python scripts/detect_tflite.py
    python scripts/detect_tflite.py --source 1 --no-smooth
"""

import argparse
import os
import sys
import time

# Add src to path (lightweight — tetap di top-level agar --help jalan
# tanpa heavy deps, lihat AGENTS.md)
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


class FPSCounter:
    def __init__(self):
        self.prev_time = 0
        self.fps = 0

    def update(self):
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        return self.fps


class ExpressionDetectorTFLite:
    def __init__(self):
        _lazy_imports()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tflite_path = os.path.join(project_root, "models/expression_model.tflite")

        if not os.path.exists(tflite_path):
            print("Model TFLite tidak ditemukan!")
            print(f"  Dicari di: {tflite_path}")
            print("  Model .tflite adalah build artifact (di-gitignore).")
            print("  Jalankan dulu:  python scripts/convert_to_tflite.py")
            sys.exit(1)

        print("Memuat model TFLite...")

        # Single inference path (AGENTS.md): prediksi via EmotionPredictor
        # dengan model .tflite eksplisit (pre-check di atas memberi pesan
        # ramah bila artifact belum di-generate).
        self.predictor = EmotionPredictor(
            model_path=tflite_path,
            detection_confidence=config.DETECTION_CONFIDENCE,
        )

        print("Model TFLite dimuat!")
        print(f"Label kelas: {self.predictor.ordered_labels}")
        print(f"Face detector: {self.predictor.detector_name}")

        # Temporal smoothing (anti flicker)
        self.smoother = MultiFaceSmoother(window_size=7, ema_alpha=0.4)

        # Colors
        self.colors = config.COLORS

    def draw_results(self, frame, x, y, w, h, expression, confidence):
        color = self.colors.get(expression, (255, 255, 255))

        # Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Header (background untuk label)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)

        # Text - gunakan warna hitam agar kontras dengan background
        label = expression.upper()
        cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Confidence bar
        bar_w = int(w * confidence)
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y + h), (x + bar_w, y + h + 10), color, -1)
        cv2.putText(frame, f"{int(confidence * 100)}%", (x + w - 40, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _resolve_source(self, source):
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
        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka sumber video: {video_source!r}")
            return

        fps_counter = FPSCounter()
        screenshot_count = 0

        print("\n" + "=" * 60)
        print("DETEKSI EKSPRESI REAL-TIME (TensorFlow Lite)")
        print("=" * 60)
        print("Tekan 'q' untuk keluar, 's' untuk screenshot")
        print("=" * 60 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                if not is_camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame = cv2.flip(frame, 1)
            fps = fps_counter.update()

            # Deteksi + prediksi via shared backend (terbesar dulu agar
            # slot smoother stabil; margin 10% seperti crop manual lama)
            try:
                results = self.predictor.predict_faces(frame, margin=0.1)
            except Exception as e:
                print(f"Warning prediksi gagal: {e}")
                results = []

            if use_smoothing and results:
                raw_preds = [(r["label"], r["confidence"],
                              np.array([r["probs"][lb]
                                        for lb in self.predictor.ordered_labels]),
                              self.predictor.ordered_labels) for r in results]
                smoothed = self.smoother.update(raw_preds)
                merged = [{**r, "label": lb, "confidence": cf}
                          for r, (lb, cf) in zip(results, smoothed)]
            else:
                if not use_smoothing:
                    self.smoother.reset()
                merged = results

            for r in merged:
                x, y, w, h = r["box"]
                self.draw_results(frame, x, y, w, h, r["label"], r["confidence"])

            # Info
            cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(merged)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Expression Detection (TFLite)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot disimpan: {filename}")

        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Deteksi ekspresi real-time (TFLite)")
    p.add_argument("--source", default=None,
                   help="Index webcam, path video, atau URL stream (default: config.CAMERA_INDEX)")
    p.add_argument("--no-smooth", action="store_true", help="Matikan temporal smoothing")
    return p.parse_args()


def main():
    args = parse_args()
    detector = ExpressionDetectorTFLite()
    detector.run(source=args.source, use_smoothing=not args.no_smooth)


if __name__ == "__main__":
    main()
