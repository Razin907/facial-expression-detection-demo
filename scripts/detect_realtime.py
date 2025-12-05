"""
Script untuk deteksi ekspresi wajah secara real-time menggunakan webcam
"""

import os
import sys
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modul lokal
import config
from preprocessing import (
    MediaPipeFaceDetector, 
    preprocess_face_for_prediction
)


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
        print("Memuat model dan classifier...")
        
        # Load model
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan di: {config.MODEL_PATH}")
        
        # Load model dengan compile=False untuk menghindari error kompatibilitas
        self.model = load_model(config.MODEL_PATH, compile=False)
        print(f"Model dimuat dari: {config.MODEL_PATH}")
        
        # Load labels
        self.class_labels = config.DEFAULT_LABELS
        print(f"Label kelas: {list(self.class_labels.values())}")
        
        # Load face detector (MediaPipe)
        self.face_detector = MediaPipeFaceDetector(
            min_detection_confidence=config.DETECTION_CONFIDENCE
        )
        print("Face detector (MediaPipe) dimuat")
        
        # Colors
        self.colors = config.COLORS
    
    def predict_expression(self, face_image):
        """
        Prediksi ekspresi dari gambar wajah
        """
        # Pra-pemrosesan
        processed_face = preprocess_face_for_prediction(face_image)
        
        # Prediksi
        predictions = self.model.predict(processed_face, verbose=0)
        
        # Ambil kelas dengan probabilitas tertinggi
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Konversi index ke label
        expression_label = self.class_labels.get(str(class_idx), 'unknown')
        
        return expression_label, confidence
    
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
    
    def run(self, camera_index=config.CAMERA_INDEX):
        """
        Jalankan deteksi real-time dari webcam
        """
        print(f"\nMembuka kamera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
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
            
            # Deteksi wajah (MediaPipe)
            faces = self.face_detector.detect_faces(frame)
            
            # Proses setiap wajah
            for (x, y, w, h) in faces:
                # Crop wajah dengan margin sedikit agar tidak terlalu ketat
                # Margin 10%
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                
                x_start = max(0, x - margin_x)
                y_start = max(0, y - margin_y)
                x_end = min(frame.shape[1], x + w + margin_x)
                y_end = min(frame.shape[0], y + h + margin_y)
                
                face_roi = frame[y_start:y_end, x_start:x_end]
                
                if face_roi.size == 0:
                    continue
                
                # Prediksi ekspresi
                expression, confidence = self.predict_expression(face_roi)
                
                # Gambar hasil
                self.draw_results(frame, x, y, w, h, expression, confidence)
            
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
                frame, f"Faces: {len(faces)}", (10, 60),
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
    try:
        detector = ExpressionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
