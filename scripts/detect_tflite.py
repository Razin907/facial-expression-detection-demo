"""
Deteksi ekspresi wajah real-time dengan TensorFlow Lite (lebih cepat!)
"""

import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from preprocessing import MediaPipeFaceDetector, preprocess_face_for_prediction


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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tflite_path = os.path.join(project_root, "models/expression_model.tflite")
        
        if not os.path.exists(tflite_path):
            print("Model TFLite tidak ditemukan!")
            print("Jalankan dulu: python convert_to_tflite.py")
            sys.exit(1)
        
        print("Memuat model TFLite...")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model TFLite dimuat!")
        print(f"Input shape: {self.input_details[0]['shape']}")
        
        # Labels
        self.class_labels = config.DEFAULT_LABELS
        
        # Face detector
        self.face_detector = MediaPipeFaceDetector(
            min_detection_confidence=config.DETECTION_CONFIDENCE
        )
        
        # Colors
        self.colors = config.COLORS
    
    def predict(self, face_image):
        try:
            # Preprocess
            processed = preprocess_face_for_prediction(face_image)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed.astype(np.float32))
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            # Convert integer to string for dictionary lookup
            expression = self.class_labels.get(str(class_idx), "unknown")
            
            return expression, confidence
        except Exception as e:
            print(f"Error predict: {e}")
            return "netral", 0.0
    
    def draw_results(self, frame, x, y, w, h, expression, confidence):
        color = self.colors.get(expression, (255, 255, 255))
        
        # Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Header (background untuk label)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        
        # Text - gunakan warna hitam agar kontras dengan background
        label = expression.upper()
        cv2.putText(frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Confidence bar
        bar_w = int(w * confidence)
        cv2.rectangle(frame, (x, y+h), (x+w, y+h+10), (50,50,50), -1)
        cv2.rectangle(frame, (x, y+h), (x+bar_w, y+h+10), color, -1)
        cv2.putText(frame, f"{int(confidence*100)}%", (x+w-40, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_counter = FPSCounter()
        
        print("\n" + "="*60)
        print("DETEKSI EKSPRESI REAL-TIME (TensorFlow Lite)")
        print("="*60)
        print("Tekan 'q' untuk keluar")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            fps = fps_counter.update()
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                margin = int(w * 0.1)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                expression, confidence = self.predict(face_roi)
                self.draw_results(frame, x, y, w, h, expression, confidence)
            
            # Info
            cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Expression Detection (TFLite)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = ExpressionDetectorTFLite()
    detector.run()


if __name__ == "__main__":
    main()
