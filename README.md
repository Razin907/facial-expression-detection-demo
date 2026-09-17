# Deteksi Ekspresi Wajah Real-time dengan CNN

Sistem deteksi ekspresi wajah menggunakan Convolutional Neural Network (CNN) yang sudah dilatih dan siap pakai. Model dapat mendeteksi 7 ekspresi wajah: **marah**, **jijik**, **takut**, **senang**, **netral**, **sedih**, dan **kaget**.

## Fitur :

- **Langsung Pakai** - Model sudah dilatih, tidak perlu training lagi!
- **Real-time Detection** - Deteksi ekspresi dari webcam dengan **MediaPipe**
- **TensorFlow Lite** - Versi ringan untuk performa lebih cepat
- **Multi-face Support** - Dapat mendeteksi beberapa wajah sekaligus
- **Screenshot Support** - Simpan hasil deteksi dengan tekan 's'

## Struktur Project

```
ekspresi-wajah-demo/
├── src/                    # Source code utama
│   ├── config.py           # Konfigurasi aplikasi
│   ├── model.py            # Arsitektur CNN
│   ├── preprocessing.py    # Preprocessing data
│   ├── smoothing.py        # Temporal smoothing anti-flicker
│   ├── inference.py        # Helper inference (.h5/.tflite) untuk Web & API
│   └── api.py              # FastAPI (REST + WebSocket)
├── app.py                  # Web demo (Streamlit)
├── Dockerfile              # Container API (port 8000) / Streamlit (8501)
├── static/realtime.html      # Halaman webcam realtime (via API /realtime)
├── scripts/                # Script yang bisa dijalankan
│   ├── train.py            # Training model
│   ├── evaluate.py         # Evaluasi: report + confusion matrix -> reports/
│   ├── detect_realtime.py  # Deteksi real-time (Keras)
│   ├── detect_tflite.py    # Deteksi real-time (TFLite - lebih cepat)
│   ├── convert_to_tflite.py# Konversi model ke TFLite
│   ├── ar_filter.py          # Filter emoji AR (EmotionPredictor)
│   └── download_assets.py    # Unduh aset emoji filter
├── tests/                  # Testing
│   ├── test_setup.py       # Verifikasi instalasi (manual)
│   ├── test_smoothing.py   # Unit test smoother (unittest)
│   └── test_labels.py      # Unit test label map (unittest)
├── notebooks/              # Jupyter notebooks
│   └── Kaggle_Training.ipynb
├── models/                 # .h5/.tflite build artifact (di-gitignore);
│                              # class_labels.json + Haar ter-track
├── dataset/                # Dataset (di-gitignore)
├── requirements.txt
├── AGENTS.md              # Panduan agen (single-path, lazy import)
└── README.md
```

## Quick Start 
### 1. Clone Repository

```powershell
git clone https://github.com/Razin907/facial-expression-detection-demo.git
cd facial-expression-detection-demo
```

### 2. Install Dependencies

```powershell
# Buat virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi!

```powershell
# Versi TFLite (Lebih Cepat - Recommended!)
python scripts/detect_tflite.py

# Atau versi Keras (Lebih Lengkap)
python scripts/detect_realtime.py

# Semua script deteksi/AR mendukung --source dan --no-smooth:
# python scripts/detect_realtime.py --source 1
# python scripts/detect_tflite.py --source video.mp4 --no-smooth
# python scripts/ar_filter.py --source 0
```

> Catatan: `models/*.tflite` adalah build artifact (di-gitignore) dan
> tidak ikut ter-clone. Generate dengan `python scripts/convert_to_tflite.py`.
>
> Butuh `tensorflow>=2.21` (converter 2.17 crash untuk model Keras 3).
> Untuk fidelitas maksimal (float32, ~17MB):
> `python scripts/convert_to_tflite.py --quantize none`.

## Evaluasi Model

```powershell
python scripts/evaluate.py
# Output: reports/metrics.json + reports/confusion_matrix.png
```

## Web Demo & API

```powershell
# Web demo: upload gambar + foto kamera + video realtime (WebRTC)
streamlit run app.py
# (tab Realtime butuh: pip install streamlit-webrtc)

# REST + WebSocket API + halaman webcam realtime (tanpa install tambahan)
uvicorn src.api:app --host 0.0.0.0 --port 8000
# GET  /health | POST /predict/image | WS /ws/predict
# Buka http://localhost:8000/realtime untuk deteksi realtime di browser

# Via Docker
# docker build -t ekspresi-wajah .
# docker run -p 8000:8000 -v ./models:/app/models ekspresi-wajah
```

## Cara Penggunaan

**Kontrol:**
- Tekan **`q`** untuk keluar
- Tekan **`s`** untuk screenshot (disimpan sebagai `screenshot_*.jpg`)

**Output di Layar:**
- Kotak berwarna di sekitar wajah (Warna berbeda tiap ekspresi)
- Label ekspresi + confidence score (Persentase keyakinan)
- Jumlah wajah terdeteksi
- FPS (Frame per second)

## Label & Warna Ekspresi

| Ekspresi | Label | Warna |
|----------|-------|-------|
| 😠 Marah | `marah` | 🔴 Merah |
| 🤢 Jijik | `jijik` | Teal (toska) |
| 😨 Takut | `takut` | 🟣 Ungu |
| 😊 Senang | `senang` | 🟢 Hijau |
| 😐 Netral | `netral` | ⚪ Putih |
| 😢 Sedih | `sedih` | 🔵 Biru |
| 😲 Kaget | `kaget` | 🟡 Kuning |

## 🔧 Troubleshooting

### Kamera tidak terbuka
**Solusi:**
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain (Zoom, Teams, dll).
- Coba ubah `CAMERA_INDEX=0` menjadi `CAMERA_INDEX=1` di `src/config.py`.

### Error "Model tidak ditemukan"
**Solusi:**
- Jalankan dari folder project (CWD = repo root).
- `EmotionPredictor` otomatis memakai `.tflite` bila ada, fallback ke `.h5`;
  pastikan minimal satu ada di `models/`. Generate `.tflite` dengan
  `python scripts/convert_to_tflite.py`.

### Import Error / Module Not Found
**Solusi:**
- Pastikan virtual environment aktif (`(.venv)` muncul di terminal).
- Jalankan ulang `pip install -r requirements.txt`.

### Convert TFLite gagal (TypeError di tflite_keras_util)
**Solusi:**
- Upgrade: `pip install -r requirements.txt` (wajib `tensorflow>=2.21`).
- Jangan set `TF_USE_LEGACY_KERAS=1` untuk model format Keras 3.
- Script otomatis fallback: from_keras_model -> SavedModel -> concrete function.

---

## Development (Opsional)

Bagian ini hanya untuk Anda yang ingin mengembangkan ulang atau melatih model sendiri.

<details>
<summary>klik untuk melihat detail training</summary>

### Persiapan Dataset
1. Download dataset **FER-2013** dari Kaggle.
2. Ekstrak ke folder `dataset/train` dan `dataset/validation`.

### Training Ulang
Jalankan perintah ini untuk melatih model baru:
```powershell
python scripts/train.py
```
Model baru akan disimpan di `models/expression_model.h5`.

### Konversi ke TFLite
Untuk performa lebih cepat, konversi model ke TFLite:
```powershell
python scripts/convert_to_tflite.py
```
</details>

---

## Author

**Razin907**
- GitHub: [@Razin907](https://github.com/Razin907)
- Repository: [facial-expression-detection-demo](https://github.com/Razin907/facial-expression-detection-demo)

## Contributing

Kontribusi sangat diterima! Silakan buka [Issues](https://github.com/Razin907/facial-expression-detection-demo/issues) atau [Pull Request](https://github.com/Razin907/facial-expression-detection-demo/pulls).
