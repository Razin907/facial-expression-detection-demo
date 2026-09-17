# AGENTS.md — facial-expression-detection-demo

## Run from repo root; scripts self-wire `src` via `sys.path`
All entrypoints assume CWD = repo root. `scripts/*` do `sys.path.insert(.../src)` themselves — do not `cd scripts/`.
- Desktop (OpenCV GUI): `python scripts/detect_realtime.py` (Keras) / `python scripts/detect_tflite.py` (faster)
- All detection scripts accept `--source [index|path|URL]` and `--no-smooth`
- Web: `streamlit run app.py` · API: `uvicorn src.api:app --host 0.0.0.0 --port 8000` (open `http://localhost:8000/realtime`, never `0.0.0.0` in a browser)
- Eval: `python scripts/evaluate.py` → `reports/metrics.json` + `confusion_matrix.png`
- Train: `python scripts/train.py` · Convert: `python scripts/convert_to_tflite.py [--quantize float16|int8]`

## Environment gotchas
- `.venv/` is a **Windows** venv (`Scripts/`, `Lib/`) — unusable from WSL/Linux shells. Verify with `python3 -m py_compile <files>` when deps (TF/cv2/mediapipe) aren't installed; full runtime needs a native env + `pip install -r requirements.txt`.
- Heavy deps (TF, matplotlib) are imported **lazily** inside functions so `--help` works without them — keep it that way when adding scripts.
- Set `TF_CPP_MIN_LOG_LEVEL=2` before any TF import (already done in `app.py`, `src/api.py`) to silence oneDNN/CPU warnings.

## Data & model artifacts (all gitignored)
- `dataset/` and `models/*.h5|*.tflite` are absent on fresh clone. `dataset/train|validation` need 7 subfolders named exactly like the training classes (Indonesian: `marah jijik takut senang netral sedih kaget`), else label mapping breaks.
- Missing `.tflite` is expected — it's a build artifact. Generate via `convert_to_tflite.py`, don't commit binaries.
- `models/haarcascade_frontalface_default.xml` is the Haar fallback asset — don't delete.

## MediaPipe version trap
`mp.solutions` was removed in mediapipe ≥ 0.10.30. Never use `mp.solutions.*` directly — always go through `create_face_detector()` in `src/preprocessing.py` (MediaPipe when available, auto-fallback to Haar). `requirements.txt` intentionally leaves mediapipe unpinned.

## Architecture (single inference path — reuse it)
- `src/inference.py::EmotionPredictor` is the shared backend for Streamlit + FastAPI (auto picks `.tflite` if present, else `.h5`). New interfaces must reuse it, not reimplement predict loops.
- Temporal smoothing lives in `src/smoothing.py` (`MultiFaceSmoother`); faces are sorted largest-first so smoother slots stay stable.
- `src/config.py` switches Kaggle vs local paths by detecting `/kaggle/input` — keep path logic there, not in scripts.

## Repo quirks
- `README.md` uses **CRLF** line endings — the Edit tool can't match its strings; patch via `python3` byte-level replace instead.
- `git status` may show whole-file diffs on untouched files (line-ending noise) — check `git diff` for your own files only before concluding anything.
- No pytest/CI/lint config exists; `tests/test_setup.py` is a manual checklist script, not a test suite.
