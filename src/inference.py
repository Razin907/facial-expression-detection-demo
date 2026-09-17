"""
Helper inference bersama untuk Streamlit app & FastAPI.

Mendukung model Keras (.h5) dan TFLite (.tflite), serta deteksi
wajah MediaPipe (fallback otomatis ke Haar Cascade bila tak tersedia). Output per wajah: box + label + confidence + probs.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from preprocessing import create_face_detector, preprocess_face_for_prediction


class EmotionPredictor:
    """Prediktor emosi yang agnostik terhadap backend model."""

    def __init__(self, model_path=None, detection_confidence=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_h5 = os.path.join(project_root, "models/expression_model.h5")
        default_tflite = os.path.join(project_root, "models/expression_model.tflite")

        if model_path is None:
            # Preferensi: TFLite (lebih cepat) bila ada, fallback ke .h5
            model_path = default_tflite if os.path.exists(default_tflite) else default_h5

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}. "
                "Latih dengan scripts/train.py lalu konversi "
                "dengan scripts/convert_to_tflite.py"
            )

        self.model_path = model_path
        self.is_tflite = model_path.endswith(".tflite")
        self.class_labels = config.load_labels()
        self.ordered_labels = [self.class_labels.get(str(i), "netral") for i in range(7)]

        if self.is_tflite:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path, compile=False)

        self.detector = create_face_detector(
            min_detection_confidence=(
                config.DETECTION_CONFIDENCE
                if detection_confidence is None else detection_confidence
            )
        )
        self.detector_name = type(self.detector).__name__

    def _predict_probs(self, face_roi):
        processed = preprocess_face_for_prediction(face_roi)
        if self.is_tflite:
            self.interpreter.set_tensor(
                self.input_details[0]["index"], processed.astype(np.float32)
            )
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            probs = self.model.predict(processed, verbose=0)[0]
        return np.array(probs, dtype=float)

    def predict_faces(self, image_bgr, margin=0.1):
        """Deteksi + prediksi semua wajah pada gambar BGR.

        Returns: list of dict(box, label, confidence, probs{label: float}).
        """
        faces = self.detector.detect_faces(image_bgr)
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        results = []
        h_img, w_img = image_bgr.shape[:2]
        for (x, y, w, h) in faces:
            mx, my = int(w * margin), int(h * margin)
            x1, y1 = max(0, x - mx), max(0, y - my)
            x2, y2 = min(w_img, x + w + mx), min(h_img, y + h + my)
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            probs = self._predict_probs(roi)
            idx = int(np.argmax(probs))
            label = self.class_labels.get(str(idx), "unknown")
            results.append({
                "box": [int(x), int(y), int(w), int(h)],
                "label": label,
                "confidence": float(probs[idx]),
                "probs": {lb: float(p) for lb, p in zip(self.ordered_labels, probs)},
            })
        return results

    def annotate(self, image_bgr, results):
        """Gambar box + label pada salinan gambar."""
        out = image_bgr.copy()
        colors = config.COLORS
        for r in results:
            x, y, w, h = r["box"]
            color = colors.get(r["label"], (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(out, (x, y - 28), (x + w, y), color, -1)
            cv2.putText(out, f"{r['label'].upper()} {r['confidence'] * 100:.0f}%",
                        (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out
