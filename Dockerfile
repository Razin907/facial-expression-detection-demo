# Demo desktop (OpenCV GUI) + Web/API tanpa GPU.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependensi sistem untuk opencv + mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY app.py ./
# Aset tracked yang wajib ada di image (label map + Haar fallback —
# lihat AGENTS.md; keduanya ter-commit, jadi aman di fresh clone).
COPY models/class_labels.json ./models/class_labels.json
COPY models/haarcascade_frontalface_default.xml ./models/haarcascade_frontalface_default.xml
# Model .h5/.tflite di-mount saat run (build artifact, di-gitignore):
#   docker run -v ./models:/app/models ...

EXPOSE 8000 8501

# Default: API. Untuk Streamlit, override CMD.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
