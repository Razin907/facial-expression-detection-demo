"""
Script untuk pra-pemrosesan data dan augmentasi
"""

import os
import cv2
import numpy as np

# NOTE: `mediapipe` dan `tensorflow` diimport lazy di dalam fungsi/class.
# Alasan: AGENTS.md mewajibkan heavy deps (TF/matplotlib) lazy agar
# `--help` jalan tanpa install; plus MediaPipe >= 0.10.30 menghapus
# legacy API `mp.solutions`, sehingga import top-level harus dihindari.


def _get_image_data_generator():
    """Import lazy ImageDataGenerator (butuh TF hanya saat training/eval)."""
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "tensorflow belum terinstall. Jalankan: pip install -r requirements.txt"
        ) from e
    return tf.keras.preprocessing.image.ImageDataGenerator


def _load_mediapipe_solutions():
    """Import mediapipe + kembalikan modul solutions, atau raise pesan jelas."""
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "mediapipe belum terinstall. Jalankan: pip install mediapipe"
        ) from e
    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "face_detection"):
        version = getattr(mp, "__version__", "unknown")
        raise AttributeError(
            f"mediapipe v{version} tidak lagi menyediakan `mp.solutions` "
            "(dihapus sejak 0.10.30). Pilih salah satu:\n"
            '  1. Otomatis pakai fallback Haar Cascade (tanpa install apa pun), via create_face_detector()\n'
            '  2. Downgrade: pip install "mediapipe==0.10.21"\n'
            "  3. Migrasi ke MediaPipe Tasks API (butuh file model .task)"
        )
    return solutions


class MediaPipeFaceDetector:
    """
    Face detector using MediaPipe (legacy Solutions API, mediapipe < 0.10.30).

    Untuk kode yang tahan versi, gunakan create_face_detector() yang otomatis
    fallback ke Haar Cascade bila `mp.solutions` tidak tersedia.
    """
    def __init__(self, min_detection_confidence=0.5):
        solutions = _load_mediapipe_solutions()
        self.mp_face_detection = solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of faces [(x, y, w, h), ...]
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure coordinates are within bounds
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                faces.append((x, y, width, height))

        return faces


class HaarFaceDetector:
    """
    Face detector menggunakan Haar Cascade (OpenCV bawaan).

    Interface sama dengan MediaPipeFaceDetector: detect_faces(image BGR)
    -> [(x, y, w, h), ...]. Dipakai sebagai fallback otomatis bila
    MediaPipe tidak tersedia / versinya tanpa `mp.solutions`.
    """
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        self.face_cascade = load_haarcascade()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def create_face_detector(min_detection_confidence=0.5, prefer="mediapipe"):
    """Factory detektor wajah yang tahan versi MediaPipe.

    - Coba MediaPipe (legacy Solutions API) bila diminta & tersedia.
    - Jika gagal (mis. mediapipe >= 0.10.30 tanpa `mp.solutions`),
      otomatis fallback ke Haar Cascade dengan pesan peringatan.

    Returns: objek dengan metode detect_faces(image) -> [(x, y, w, h), ...].
    """
    if prefer == "mediapipe":
        try:
            detector = MediaPipeFaceDetector(
                min_detection_confidence=min_detection_confidence
            )
            print("Face detector: MediaPipe")
            return detector
        except Exception as e:
            print(f"Warning: MediaPipe tidak bisa dipakai ({e})")
            print("Fallback ke Haar Cascade...")
    detector = HaarFaceDetector()
    print("Face detector: Haar Cascade")
    return detector


def create_data_generators(train_dir, validation_dir, target_size=(48, 48), 
                           batch_size=64, augmentation=True):
    """
    Membuat data generator untuk training dan validation
    
    Args:
        train_dir: Path ke folder training
        validation_dir: Path ke folder validation
        target_size: Ukuran target gambar (height, width)
        batch_size: Ukuran batch untuk training
        augmentation: Apakah menggunakan augmentasi atau tidak
        
    Returns:
        train_generator, validation_generator
    """
    ImageDataGenerator = _get_image_data_generator()

    if augmentation:
        # Data Generator untuk Training dengan Augmentasi AGRESIF
        # Augmentasi lebih kuat untuk improve generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,                    # Normalisasi pixel [0, 255] -> [0, 1]
            rotation_range=20,                 # ↑ Rotasi acak hingga 20 derajat (dari 15)
            width_shift_range=0.2,             # ↑ Geser horizontal 20% (dari 15%)
            height_shift_range=0.2,            # ↑ Geser vertikal 20% (dari 15%)
            shear_range=0.2,                   # ↑ Transformasi shear 20% (dari 15%)
            zoom_range=0.2,                    # ↑ Zoom in/out 20% (dari 15%)
            brightness_range=[0.7, 1.3],       # ✨ NEW: Random brightness untuk robust terhadap lighting
            horizontal_flip=True,              # Balik horizontal
            fill_mode='nearest'                # Mengisi pixel yang kosong
        )
    else:
        # Tanpa augmentasi, hanya normalisasi
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Data Generator untuk Validation (hanya normalisasi, tanpa augmentasi)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',              # Grayscale untuk ekspresi wajah
        class_mode='categorical',            # Multi-class classification
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False                        # Tidak perlu shuffle untuk validation
    )
    
    return train_generator, validation_generator


def preprocess_face_for_prediction(face_image, target_size=(48, 48)):
    """
    Pra-pemrosesan gambar wajah untuk prediksi
    
    Args:
        face_image: Gambar wajah (numpy array)
        target_size: Ukuran target (height, width)
        
    Returns:
        Gambar yang sudah diproses dan siap untuk prediksi
    """
    # Ubah ke grayscale jika belum
    if len(face_image.shape) == 3:
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_image
    
    # ✨ Histogram Equalization untuk meningkatkan kontras dan akurasi
    # Ini sangat membantu di kondisi lighting yang kurang optimal
    face_gray = cv2.equalizeHist(face_gray)
    
    # Resize ke ukuran target
    face_resized = cv2.resize(face_gray, target_size)
    
    # Normalisasi [0, 255] -> [0, 1]
    face_normalized = face_resized / 255.0
    
    # Reshape untuk input model: (1, height, width, 1)
    face_processed = face_normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return face_processed


def load_haarcascade():
    """
    Memuat Haar Cascade classifier untuk deteksi wajah

    Prioritas: aset tracked `models/haarcascade_frontalface_default.xml`
    (jangan dihapus — lihat AGENTS.md), fallback ke bawaan OpenCV.
    
    Returns:
        Face cascade classifier
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracked = os.path.join(base_dir, "models", "haarcascade_frontalface_default.xml")
    if os.path.exists(tracked):
        cascade_path = tracked
    else:
        # Fallback ke bawaan OpenCV bila aset tracked absen
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise IOError("Tidak dapat memuat Haar Cascade untuk deteksi wajah")
    
    return face_cascade


def detect_faces(image, face_cascade=None, scale_factor=1.05, min_neighbors=4, use_mediapipe=False):
    """
    Mendeteksi wajah dalam gambar
    
    Args:
        image: Gambar input
        face_cascade: Haar Cascade classifier (optional if using MediaPipe)
        scale_factor: Parameter untuk Haar Cascade
        min_neighbors: Parameter untuk Haar Cascade
        use_mediapipe: Boolean to use MediaPipe instead of Haar Cascade
        
    Returns:
        List koordinat wajah [(x, y, w, h), ...]
    """
    if use_mediapipe:
        detector = MediaPipeFaceDetector()
        return detector.detect_faces(image)
        
    # Fallback to Haar Cascade
    if face_cascade is None:
        raise ValueError("face_cascade must be provided if not using MediaPipe")
        
    # Convert ke grayscale jika belum
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    return faces


def print_dataset_info(train_generator, validation_generator):
    """
    Menampilkan informasi dataset
    """
    print("\n" + "=" * 70)
    print("INFORMASI DATASET")
    print("=" * 70)
    print(f"Jumlah data training: {train_generator.n}")
    print(f"Jumlah data validation: {validation_generator.n}")
    print(f"Jumlah kelas: {train_generator.num_classes}")
    print(f"Nama kelas: {list(train_generator.class_indices.keys())}")
    print(f"Ukuran batch: {train_generator.batch_size}")
    print(f"Ukuran gambar: {train_generator.target_size}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # Test preprocessing
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'validation')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Membuat data generators...")
        train_gen, val_gen = create_data_generators(train_dir, val_dir)
        print_dataset_info(train_gen, val_gen)
    else:
        print("Dataset belum tersedia. Silakan tambahkan data ke folder dataset/")
    
    # Test Haar Cascade
    print("Testing Haar Cascade untuk deteksi wajah...")
    try:
        face_cascade = load_haarcascade()
        print("✓ Haar Cascade berhasil dimuat!")
    except Exception as e:
        print(f"✗ Error: {e}")
