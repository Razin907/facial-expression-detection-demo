"""
Script untuk mengkonversi model Keras (.h5) ke TensorFlow Lite (.tflite)

Catatan: *.tflite adalah build artifact (di-gitignore), jadi file ini
memang tidak ada setelah fresh clone. Generate dengan:

    python scripts/convert_to_tflite.py
    python scripts/convert_to_tflite.py --quantize float16
"""

import argparse
import os
import sys

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

H5_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/expression_model.h5")
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/expression_model.tflite")


class _NeedConcreteFunction(Exception):
    """Sinyal internal: export SavedModel gagal, lanjut ke Lapis 3."""


def convert_to_tflite(h5_path=H5_MODEL_PATH, tflite_path=TFLITE_MODEL_PATH,
                      quantize=None):
    # Import berat lazy agar --help jalan tanpa TF (lihat AGENTS.md)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if not os.path.exists(h5_path):
        print(f"Model tidak ditemukan: {h5_path}")
        print("Latih dulu dengan: python scripts/train.py")
        sys.exit(1)

    print(f"Loading model: {h5_path}")

    # Load model. CATATAN: file .h5 format Keras 3 (mis. dari notebook
    # Kaggle) hanya bisa dibaca Keras 3 — JANGAN set TF_USE_LEGACY_KERAS=1
    # karena tf_keras legacy gagal di InputLayer 'batch_shape'.
    # Tanpa env var: load sukses (Keras 3), convert jatuh ke Lapis 2.
    try:
        model = tf.keras.models.load_model(h5_path, compile=False)
    except TypeError as e:
        print(f"Gagal load model ({e}).")
        if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
            print("Penyebab: TF_USE_LEGACY_KERAS=1 memaksa tf_keras legacy,")
            print("padahal .h5 ini format Keras 3. Jalankan ulang TANPA env var:")
            print("  Remove-Item Env:TF_USE_LEGACY_KERAS")
            print("  python scripts/convert_to_tflite.py")
            sys.exit(2)
        raise
    model.summary()

    def _configure(converter):
        # Optimasi kecepatan; setting kuantisasi dipakai kedua jalur.
        # "none" = float32 murni (paling setia ke .h5, file ~17MB).
        if quantize == "none":
            print("Kuantisasi: none (float32, fidelitas maksimal)")
            return converter
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]
            print("Kuantisasi: float16")
        elif quantize == "int8":
            # Post-training int8 quantization (perlu representative dataset
            # untuk hasil optimal; di sini memakai optimasi default saja).
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            print("Kuantisasi: int8 (default calibration)")
        return converter

    # Convert to TFLite
    print("\nConverting to TFLite...")
    try:
        # Lapis 1: from_keras_model. Di TF 2.16+ jalur ini butuh legacy
        # tf_keras (pip install tf_keras + TF_USE_LEGACY_KERAS=1).
        # Model Keras 3 (mis. dari notebook Kaggle) gagal di sini dengan
        # TypeError di tflite_keras_util -> jatuh ke Lapis 2.
        converter = _configure(tf.lite.TFLiteConverter.from_keras_model(model))
        tflite_model = converter.convert()
        print("Konversi via from_keras_model OK")
    except TypeError as e:
        print(f"from_keras_model gagal ({e}).")
        print("Fallback: export SavedModel dulu (untuk model Keras 3)...")
        export_fn = getattr(model, "export", None)
        if export_fn is None:
            print("Model ini tidak mendukung .export() — tidak bisa convert.")
            print("Jalan bersih: retrain dengan python scripts/train.py")
            sys.exit(2)
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="savedmodel_")
        try:
            export_path = os.path.join(tmp_dir, "saved_model")
            try:
                export_fn(export_path)
            except TypeError as e2:
                # Diketahui di TF 2.17 + Keras 3.14: export crash di
                # trackable converter ('_DictWrapper'). Lanjut ke Lapis 3.
                print(f"Export SavedModel gagal ({e2}).")
                raise _NeedConcreteFunction from e2
            converter = _configure(
                tf.lite.TFLiteConverter.from_saved_model(export_path)
            )
            tflite_model = converter.convert()
            print("Konversi via SavedModel OK")
        except _NeedConcreteFunction:
            # Lapis 3: trace manual via tf.function. Melewati
            # tflite_keras_util._wrapped_model (rusak utk Keras 3 di
            # TF 2.17) karena converter menerima concrete function jadi.
            print("Fallback terakhir: trace via concrete function...")
            in_shape = list(model.input_shape)
            in_shape[0] = 1  # batch statis untuk TFLite
            infer = tf.function(
                lambda x: model(x, training=False),
                input_signature=[tf.TensorSpec(in_shape, tf.float32)],
            )
            concrete = infer.get_concrete_function()
            try:
                converter = _configure(
                    tf.lite.TFLiteConverter.from_concrete_functions([concrete])
                )
                tflite_model = converter.convert()
            except Exception as e3:
                print(f"Konversi concrete function gagal ({e3}).")
                print("Semua jalur convert gugur — jalan bersih: retrain")
                print("dengan python scripts/train.py (format tf.keras env ini).")
                sys.exit(2)
            print("Konversi via concrete function OK")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Verifikasi cepat: interpreter bisa allocate + input shape benar
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_shape = interpreter.get_input_details()[0]['shape']
    print(f"Verifikasi OK, input shape: {input_shape}")

    # Save
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\nModel TFLite tersimpan di: {tflite_path}")

    # Size comparison
    h5_size = os.path.getsize(h5_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print("\nUkuran file:")
    print(f"  .h5:     {h5_size:.2f} MB")
    print(f"  .tflite: {tflite_size:.2f} MB")
    print(f"  Kompresi: {(1 - tflite_size / h5_size) * 100:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Konversi model Keras ke TFLite")
    p.add_argument("--model", default=H5_MODEL_PATH, help="Path model .h5 input")
    p.add_argument("--output", default=TFLITE_MODEL_PATH, help="Path .tflite output")
    p.add_argument("--quantize", choices=["none", "float16", "int8"], default=None,
                   help="Mode kuantisasi (default: optimasi standar; "
                        "'none' = float32, paling akurat)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_tflite(h5_path=args.model, tflite_path=args.output,
                      quantize=args.quantize)
