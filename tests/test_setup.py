"""
Script untuk testing dan verifikasi instalasi
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config

def test_imports():
    """Test import semua library yang dibutuhkan"""
    print("TESTING LIBRARY IMPORTS")
    
    tests = [
        ("TensorFlow", "tensorflow"),
        ("Keras", "tensorflow.keras"),
        ("OpenCV", "cv2"),
        ("MediaPipe", "mediapipe"),
        ("NumPy", "numpy"),
        ("Matplotlib", "matplotlib"),
        ("Scikit-learn", "sklearn"),
    ]
    
    failed = []
    
    for name, module in tests:
        try:
            __import__(module)
            version = ""
            if module == "tensorflow":
                import tensorflow as tf
                version = f" (v{tf.__version__})"
            elif module == "cv2":
                import cv2
                version = f" (v{cv2.__version__})"
            elif module == "mediapipe":
                import mediapipe as mp
                version = f" (v{mp.__version__})"
            elif module == "numpy":
                import numpy as np
                version = f" (v{np.__version__})"
            
            print(f"[OK] {name:20s} {version}")
        except ImportError as e:
            print(f"[FAIL] {name:20s} - NOT INSTALLED")
            failed.append(name)
    
    print("=" * 70)
    
    if failed:
        print(f"\nFAILED: {len(failed)} library tidak terinstall:")
        for lib in failed:
            print(f"   - {lib}")
        print("\nJalankan: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] SUKSES: Semua library terinstall dengan benar!")
        return True


def test_gpu():
    """Test apakah GPU tersedia untuk TensorFlow"""
    print("\n" + "=" * 70)
    print("TESTING GPU AVAILABILITY")
    print("=" * 70)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"[OK] GPU terdeteksi: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print("\n* Training akan menggunakan GPU (lebih cepat)")
        else:
            print("[INFO] Tidak ada GPU terdeteksi")
            print("* Training akan menggunakan CPU (lebih lambat)")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error checking GPU: {e}")
        return False


def test_camera():
    """Test apakah OpenCV dapat mengakses kamera"""
    print("\n" + "=" * 70)
    print("TESTING CAMERA ACCESS")
    print("=" * 70)
    
    try:
        import cv2
        
        # Coba buka kamera sesuai config
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"[FAIL] Kamera index {config.CAMERA_INDEX} tidak dapat diakses")
            print("* Pastikan webcam terhubung dan tidak digunakan aplikasi lain")
            return False
        
        # Baca satu frame
        ret, frame = cap.read()
        
        if ret:
            h, w = frame.shape[:2]
            print(f"[OK] Kamera berhasil diakses")
            print(f"  Resolusi: {w}x{h}")
        else:
            print("[FAIL] Tidak dapat membaca frame dari kamera")
            cap.release()
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_mediapipe():
    """Test apakah MediaPipe Face Detection berfungsi"""
    print("\n" + "=" * 70)
    print("TESTING MEDIAPIPE FACE DETECTOR")
    print("=" * 70)
    
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Buat dummy image (hitam)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process image
        results = face_detection.process(dummy_image)
        
        print("[OK] MediaPipe Face Detection berhasil diinisialisasi")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_model_building():
    """Test apakah model dapat dibuat"""
    print("\n" + "=" * 70)
    print("TESTING MODEL BUILDING")
    print("=" * 70)
    
    try:
        from model import create_expression_model, compile_model
        
        # Buat model sederhana
        model = create_expression_model(
            input_shape=config.INPUT_SHAPE, 
            num_classes=config.NUM_CLASSES
        )
        model = compile_model(model)
        
        total_params = model.count_params()
        
        print("[OK] Model berhasil dibuat")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Input shape: {config.INPUT_SHAPE}")
        print(f"  Output classes: {config.NUM_CLASSES}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("=" * 70)
    print("                    SISTEM VERIFIKASI INSTALASI")
    print("=" * 70)
    print()
    
    results = {
        "Library Imports": test_imports(),
        "GPU Availability": test_gpu(),
        "Camera Access": test_camera(),
        "MediaPipe Detector": test_mediapipe(),
        "Model Building": test_model_building(),
    }
    
    # Summary
    print("\n")
    print("=" * 70)
    print("                        SUMMARY")
    print("=" * 70)
    print("\n")
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test:25s}: {status}")
    
    print(f"Hasil: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSEMUA TEST BERHASIL!")
        print("Anda siap untuk menjalankan aplikasi!")
        print("\nJalankan: python detect_realtime.py")
    else:
        print(f"\n{total - passed} test gagal")
        print("Silakan perbaiki masalah di atas sebelum melanjutkan")
    
    print()


if __name__ == '__main__':
    main()
