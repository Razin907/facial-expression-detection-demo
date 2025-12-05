"""
Script untuk mengkonversi model Keras (.h5) ke TensorFlow Lite (.tflite)
"""

import os
import sys
import tensorflow as tf

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

H5_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/expression_model.h5")
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/expression_model.tflite")


def convert_to_tflite():
    print(f"Loading model: {H5_MODEL_PATH}")
    
    # Load model
    model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
    model.summary()
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimasi untuk kecepatan
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nModel TFLite tersimpan di: {TFLITE_MODEL_PATH}")
    
    # Size comparison
    h5_size = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"\nUkuran file:")
    print(f"  .h5:     {h5_size:.2f} MB")
    print(f"  .tflite: {tflite_size:.2f} MB")
    print(f"  Kompresi: {(1 - tflite_size/h5_size)*100:.1f}%")


if __name__ == "__main__":
    if not os.path.exists(H5_MODEL_PATH):
        print(f"Model tidak ditemukan: {H5_MODEL_PATH}")
    else:
        convert_to_tflite()
