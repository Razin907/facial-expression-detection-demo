"""
Web demo deteksi ekspresi wajah (Streamlit).

Jalankan:
    streamlit run app.py

Fitur:
- Upload gambar (JPG/PNG)
- Foto kamera browser (st.camera_input)
- Video realtime via WebRTC (butuh `pip install streamlit-webrtc`)
- Multi-face, grafik probabilitas per wajah
"""

import os
import sys

# Redam log TensorFlow yang berisik (oneDNN, CPU instructions, deprecations).
# Harus diset SEBELUM tensorflow diimport (import terjadi lazy di inference.py).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config  # noqa: E402


def load_predictor(model_path, det_conf):
    from inference import EmotionPredictor
    return EmotionPredictor(model_path=model_path or None,
                            detection_confidence=det_conf)


def decode_upload(uploaded):
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def main():
    import streamlit as st

    st.set_page_config(page_title="Deteksi Ekspresi Wajah", layout="wide")
    st.title("Deteksi Ekspresi Wajah")
    st.caption("CNN + deteksi wajah — 7 ekspresi: marah, jijik, takut, senang, netral, sedih, kaget")

    project_root = os.path.dirname(os.path.abspath(__file__))
    default_h5 = os.path.join(project_root, "models", "expression_model.h5")
    default_tflite = os.path.join(project_root, "models", "expression_model.tflite")

    with st.sidebar:
        st.header("Pengaturan")
        model_opt = st.selectbox(
            "Model",
            ["Otomatis (TFLite bila ada)", "expression_model.h5", "expression_model.tflite", "Kustom..."],
        )
        custom_path = None
        if model_opt == "Kustom...":
            custom_path = st.text_input("Path model", value=default_h5)
        det_conf = st.slider("Detection confidence", 0.1, 0.9,
                             float(config.DETECTION_CONFIDENCE), 0.05)
        margin = st.slider("Margin crop wajah", 0.0, 0.3, 0.1, 0.05)

    if model_opt == "Otomatis (TFLite bila ada)":
        model_path = None
    elif model_opt == "expression_model.h5":
        model_path = default_h5
    elif model_opt == "expression_model.tflite":
        model_path = default_tflite
    else:
        model_path = custom_path

    @st.cache_resource(show_spinner="Memuat model...")
    def _cached_predictor(mp, dc):
        return load_predictor(mp, dc)

    try:
        predictor = _cached_predictor(model_path, det_conf)
    except FileNotFoundError as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("Latih dengan `python scripts/train.py` lalu konversi via `python scripts/convert_to_tflite.py`.")
        return
    except Exception as e:
        st.error(f"Gagal inisialisasi aplikasi: {e}")
        return

    st.caption(f"Backend: {'TFLite' if predictor.is_tflite else 'Keras'} — `{predictor.model_path}`"
               f" — Detektor: {predictor.detector_name}")

    tab_img, tab_cam, tab_live = st.tabs(["Upload Gambar", "Kamera Browser", "Realtime (WebRTC)"])

    with tab_img:
        uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = decode_upload(uploaded)
            if img is None:
                st.error("Gagal membaca gambar.")
            else:
                with st.spinner("Mendeteksi..."):
                    results = predictor.predict_faces(img, margin=margin)
                annotated = predictor.annotate(img, results)
                c1, c2 = st.columns(2)
                c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
                c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption=f"Hasil — {len(results)} wajah", use_container_width=True)
                if not results:
                    st.warning("Tidak ada wajah terdeteksi. Coba gambar lain / turunkan detection confidence.")
                else:
                    for i, r in enumerate(results):
                        with st.expander(f"Wajah {i + 1}: {r['label']} ({r['confidence'] * 100:.1f}%)"):
                            st.bar_chart(r["probs"])

    with tab_cam:
        st.write("Ambil foto dari kamera browser:")
        photo = st.camera_input("Kamera")
        if photo is not None:
            img = decode_upload(photo)
            with st.spinner("Mendeteksi..."):
                results = predictor.predict_faces(img, margin=margin)
            annotated = predictor.annotate(img, results)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"Hasil — {len(results)} wajah", use_container_width=True)
            for i, r in enumerate(results):
                with st.expander(f"Wajah {i + 1}: {r['label']} ({r['confidence'] * 100:.1f}%)"):
                    st.bar_chart(r["probs"])

    with tab_live:
        st.write("Video realtime dari webcam (diproses frame-by-frame di server).")
        try:
            from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        except ImportError:
            st.warning("Paket `streamlit-webrtc` belum terinstall. Jalankan:")
            st.code("pip install streamlit-webrtc")
            st.info("Alternatif tanpa install: jalankan API lalu buka halaman "
                    "`http://localhost:8000/realtime` (webcam via WebSocket).")
        else:
            import av
            from smoothing import MultiFaceSmoother
            from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

            class EmotionVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    # Per-processor instance: jangan share antar sesi WebRTC
                    # (smoother di closure main() bocor state antar user/reconnect).
                    self.smoother = MultiFaceSmoother(window_size=5, ema_alpha=0.5)

                def recv(self, frame):
                    img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
                    # Satu kali inference per frame, lalu smoothing temporal
                    results = predictor.predict_faces(img, margin=margin)
                    if results:
                        raw = [(r["label"], r["confidence"],
                                np.array([r["probs"][lb] for lb in predictor.ordered_labels]),
                                predictor.ordered_labels) for r in results]
                        smoothed = self.smoother.update(raw)
                        merged = [{**r, "label": lb, "confidence": cf}
                                  for r, (lb, cf) in zip(results, smoothed)]
                        out = predictor.annotate(img, merged)
                    else:
                        self.smoother.reset()
                        out = img
                    return av.VideoFrame.from_ndarray(out, format="bgr24")

            webrtc_streamer(key="emotion-live", video_processor_factory=EmotionVideoProcessor,
                            media_stream_constraints={"video": True, "audio": False})


if __name__ == "__main__":
    main()
