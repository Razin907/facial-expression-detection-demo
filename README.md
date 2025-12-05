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
│   └── preprocessing.py    # Preprocessing data
├── scripts/                # Script yang bisa dijalankan
│   ├── train.py            # Training model
│   ├── detect_realtime.py  # Deteksi real-time (Keras)
│   ├── detect_tflite.py    # Deteksi real-time (TFLite - lebih cepat)
│   └── convert_to_tflite.py# Konversi model ke TFLite
├── tests/                  # Testing
│   └── test_setup.py       # Verifikasi instalasi
├── notebooks/              # Jupyter notebooks
│   └── Kaggle_Training.ipynb
├── models/                 # Model files (di-gitignore)
├── dataset/                # Dataset (di-gitignore)
├── requirements.txt
└── README.md
```

## Quick Start 
### 1. Clone Repository

```powershell
git clone https://github.com/Razin907/facial-expression-detection.git
cd facial-expression-detection
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
| 🤢 Jijik | `jijik` | 🟦 Teal |
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
- Pastikan file `models/expression_model.tflite` ada.
- Pastikan Anda menjalankan script dari dalam folder project.

### Import Error / Module Not Found
**Solusi:**
- Pastikan virtual environment aktif (`(.venv)` muncul di terminal).
- Jalankan ulang `pip install -r requirements.txt`.

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
- Repository: [facial-expression-detection](https://github.com/Razin907/facial-expression-detection)

## Contributing

Kontribusi sangat diterima! Silakan buka [Issues](https://github.com/Razin907/facial-expression-detection/issues) atau [Pull Request](https://github.com/Razin907/facial-expression-detection/pulls).
