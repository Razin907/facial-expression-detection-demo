"""
REST + WebSocket API untuk deteksi ekspresi wajah.

Jalankan:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoint:
- GET  /health          -> status + info model
- POST /predict/image   -> multipart image -> JSON daftar wajah
- WS   /ws/predict      -> kirim bytes JPEG per frame -> JSON per frame
- GET  /realtime        -> halaman webcam realtime (WebSocket, tanpa install)
"""

import io
import os
import sys

# Redam log TensorFlow yang berisik (oneDNN, CPU instructions).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH_ENV = os.environ.get("EXPRESSION_MODEL", None)
DET_CONF_ENV = float(os.environ.get("DETECTION_CONFIDENCE", "0.5"))

try:
    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pesan ramah bila deps API belum diinstall
    raise SystemExit(
        "Dependensi API belum terinstall. Jalankan: pip install fastapi uvicorn python-multipart"
    )

from inference import EmotionPredictor

app = FastAPI(title="Facial Expression Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Halaman demo realtime (static/realtime.html) -> GET /realtime
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATIC_DIR = os.path.join(_PROJECT_ROOT, "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/realtime")
def realtime_page():
    page = os.path.join(_STATIC_DIR, "realtime.html")
    if not os.path.exists(page):
        return {"error": "static/realtime.html tidak ditemukan"}
    return FileResponse(page, media_type="text/html")

_predictor = None

# Batas upload agar API tidak dibanjiri payload raksasa.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor(
            model_path=MODEL_PATH_ENV or None,
            detection_confidence=DET_CONF_ENV,
        )
    return _predictor


def decode_image_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.get("/", include_in_schema=False)
def root():
    # Buka root di browser -> langsung ke halaman demo realtime
    return RedirectResponse(url="/realtime")


@app.get("/health")
def health():
    try:
        p = get_predictor()
        return {"status": "ok", "backend": "tflite" if p.is_tflite else "keras",
                "model": p.model_path, "classes": p.ordered_labels,
                "detector": p.detector_name}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return {"faces": [], "error": f"tipe file {file.content_type} ditolak (kirim JPG/PNG/WebP)"}
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        return {"faces": [], "error": "file terlalu besar (maks 10MB)"}
    img = decode_image_bytes(data)
    if img is None:
        return {"faces": [], "error": "gagal decode gambar (kirim JPG/PNG valid)"}
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        return {"faces": [], "error": f"model tidak ditemukan: {e}"}
    except Exception as e:
        return {"faces": [], "error": str(e)}
    faces = predictor.predict_faces(img)
    return {"faces": faces, "count": len(faces)}


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        predictor = get_predictor()
    except Exception as e:
        try:
            await websocket.send_json({"faces": [], "error": f"model tidak tersedia: {e}"})
            await websocket.close()
        except Exception:
            pass
        return
    try:
        while True:
            data = await websocket.receive_bytes()
            img = decode_image_bytes(data)
            if img is None:
                await websocket.send_json({"faces": [], "error": "invalid jpeg bytes"})
                continue
            # Downscale ringan untuk latency (lebar maks 640)
            h, w = img.shape[:2]
            if w > 640:
                scale = 640 / w
                img = cv2.resize(img, (640, int(h * scale)))
            faces = predictor.predict_faces(img)
            await websocket.send_json({"faces": faces, "count": len(faces)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"faces": [], "error": str(e)})
            await websocket.close()
        except Exception:
            pass
