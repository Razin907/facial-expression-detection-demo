"""
Script untuk deteksi ekspresi wajah secara real-time menggunakan webcam
"""

import argparse
import os
import sys
import time

# Add project root and src to path (lightweight — tetap di top-level
# agar --help jalan tanpa install heavy deps, lihat AGENTS.md)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _lazy_imports():
    """Import berat (cv2/numpy/modul lokal) secara lazy.

    Dipanggil di awal __init__/main agar `--help` tetap jalan tanpa
    TF/cv2 terinstall. TF_CPP_MIN_LOG_LEVEL diset sebelum import TF
    (terjadi lazy di dalam EmotionPredictor).
    """
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
    """
    Class untuk menghitung FPS (Frames Per Second)
    """
    def __init__(self):
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0
        
    def update(self):
        self.curr_time = time.time()
        self.fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        return self.fps


class ExpressionDetector:
    """
    Class untuk mendeteksi ekspresi wajah secara real-time
    """
    
    def __init__(self):
        """
        Inisialisasi detector
        """
        _lazy_imports()
        print("Memuat model dan classifier...")

        # Single inference path (AGENTS.md): prediksi via EmotionPredictor.
        # Backend Keras (.h5) eksplisit agar varian script ini tidak
        # pindah ke TFLite saat artifact .tflite ada. FileNotFoundError
        # dari predictor ditangani di main().
        self.predictor = EmotionPredictor(
            model_path=config.MODEL_PATH,
            detection_confidence=config.DETECTION_CONFIDENCE,
        )
        print(f"Model dimuat dari: {self.predictor.model_path}")
        print(f"Label kelas: {self.predictor.ordered_labels}")
        print(f"Face detector: {self.predictor.detector_name}")

        # Temporal smoothing anti-flicker (satu smoother per slot wajah)
        self.smoother = MultiFaceSmoother(window_size=7, ema_alpha=0.4)

        # Colors
        self.colors = config.COLORS

    def draw_results(self, frame, x, y, w, h, expression, confidence):
        """
        Gambar kotak dan label pada frame dengan UI yang lebih modern
        """
        # Pilih warna berdasarkan ekspresi
        color = self.colors.get(expression, (255, 255, 255))
        
        # Gambar kotak dengan sudut melengkung (simulasi)
        # Karena OpenCV tidak punya rounded rectangle bawaan yang mudah, kita pakai rectangle biasa
        # tapi dengan desain yang lebih rapi
        
        # 1. Kotak utama
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # 2. Header untuk label
        header_height = 30
        cv2.rectangle(frame, (x, y-header_height), (x+w, y), color, -1)
        
        # 3. Teks Label
        label_text = expression.upper()
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Center text
        text_x = x + (w - text_w) // 2
        text_y = y - 8
        
        cv2.putText(
            frame, label_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # 4. Confidence bar di bawah kotak
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (x, y+h), (x+w, y+h+10), (50, 50, 50), -1) # Background bar
        cv2.rectangle(frame, (x, y+h), (x+bar_width, y+h+10), color, -1) # Confidence bar
        
        # 5. Teks confidence
        conf_text = f"{int(confidence*100)}%"
        cv2.putText(
            frame, conf_text, (x+w-35, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    
    def run(self, source=None, use_smoothing=True):
        """
        Menjalankan loop utama deteksi
        """
        # Resolusi --source: None -> config, digit -> int, path/URL -> str
        if source is None:
            video_source = config.CAMERA_INDEX
        elif isinstance(source, int):
            video_source = source
        else:
            s = str(source).strip()
            video_source = int(s) if s.isdigit() else s
        is_camera = isinstance(video_source, int)

        cap = cv2.VideoCapture(video_source)
        
        # Pengaturan kualitas kamera (HD 720p)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera!")
            return
        
        print("Kamera terbuka")
        print("\n" + "=" * 70)
        print("DETEKSI EKSPRESI REAL-TIME (MediaPipe)")
        print("=" * 70)
        print("Tekan 'q' untuk keluar")
        print("Tekan 's' untuk screenshot")
        print("=" * 70 + "\n")
        
        fps_counter = FPSCounter()
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera")
                break
            
            # Flip horizontal
            frame = cv2.flip(frame, 1)
            
            # Update FPS
            fps = fps_counter.update()
            
            # Deteksi + prediksi via shared backend (wajah terbesar dulu
            # agar slot smoother stabil; margin 10% seperti crop manual lama)
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

            # Gambar hasil
            for r in merged:
                x, y, w, h = r["box"]
                self.draw_results(frame, x, y, w, h, r["label"], r["confidence"])
            
            # Info UI
            # Background untuk info
            cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
            
            # FPS
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Face Count
            cv2.putText(
                frame, f"Faces: {len(merged)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Tampilkan frame
            cv2.imshow(config.WINDOW_NAME, frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nKeluar dari program...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot disimpan: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera ditutup")


def main():
    parser = argparse.ArgumentParser(description="Deteksi ekspresi real-time (Keras)")
    parser.add_argument("--source", default=None,
                        help="Index webcam, path video, atau URL stream (default: config.CAMERA_INDEX)")
    parser.add_argument("--no-smooth", action="store_true", help="Matikan temporal smoothing")
    args = parser.parse_args()
    try:
        detector = ExpressionDetector()
        detector.run(source=args.source, use_smoothing=not args.no_smooth)
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
