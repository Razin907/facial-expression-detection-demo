"""
Evaluasi model ekspresi wajah pada data validation.

Menghasilkan:
- akurasi, classification report (per-class precision/recall/f1)
- confusion matrix (confusion_matrix.png)
- ringkasan JSON (metrics.json)

Contoh:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/expression_model.h5 --validation-dir dataset/validation
"""

import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi model ekspresi wajah")
    p.add_argument("--model", default=config.MODEL_PATH, help="Path model .h5/.tflite")
    p.add_argument("--validation-dir", default=config.VALIDATION_DIR,
                   help="Folder validation (struktur flow_from_directory)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"),
        help="Folder output metrics.json + confusion_matrix.png")
    return p.parse_args()


def plot_confusion_matrix(cm, class_names, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")

    # Anotasi angka
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model tidak ditemukan: {args.model}")
        sys.exit(1)
    if not os.path.isdir(args.validation_dir):
        print(f"Error: folder validation tidak ditemukan: {args.validation_dir}")
        print("Struktur yang diharapkan: dataset/validation/<kelas>/*.jpg")
        sys.exit(1)

    # Import berat di dalam main agar --help tetap cepat
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    from preprocessing import create_data_generators

    print(f"Memuat data validation: {args.validation_dir}")
    _, val_gen = create_data_generators(
        train_dir=args.validation_dir,  # train_dir tidak dipakai isinya
        validation_dir=args.validation_dir,
        target_size=(config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]),
        batch_size=args.batch_size,
        augmentation=False,
    )
    # create_data_generators mengembalikan (train, val); kita hanya butuh val.
    # Karena train_dir == validation_dir, ambil generator kedua.
    steps = max(1, val_gen.n // val_gen.batch_size + (1 if val_gen.n % val_gen.batch_size else 0))

    if args.model.endswith(".tflite"):
        import tensorflow as tf

        print(f"Memuat model TFLite: {args.model}")
        interpreter = tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]["index"]
        output_idx = interpreter.get_output_details()[0]["index"]
        val_gen.reset()
        all_probs = []
        for _ in range(steps):
            batch_x, _ = next(val_gen)
            for sample in batch_x:
                # Interpreter TFLite expects batch dim; pakai batch=1 per sampel
                sample = np.expand_dims(sample, axis=0).astype(np.float32)
                interpreter.resize_tensor_input(input_idx, sample.shape, strict=False)
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_idx, sample)
                interpreter.invoke()
                all_probs.append(interpreter.get_tensor(output_idx)[0])
        probs = np.array(all_probs)
    else:
        from tensorflow.keras.models import load_model

        print(f"Memuat model: {args.model}")
        model = load_model(args.model, compile=False)
        val_gen.reset()
        probs = model.predict(val_gen, steps=steps, verbose=1)
    y_pred = np.argmax(probs[:val_gen.n], axis=1)
    y_true = val_gen.classes[:val_gen.n]
    idx_to_label = {v: k for k, v in val_gen.class_indices.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    acc = float(accuracy_score(y_true, y_pred))
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=class_names,
                                       zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "classes": class_names,
                   "report": report_dict}, f, indent=2)
    plot_confusion_matrix(cm, class_names,
                          os.path.join(args.output_dir, "confusion_matrix.png"))

    print("\n" + "=" * 70)
    print(f"Akurasi validation: {acc:.4f}  (n={len(y_true)})")
    print("=" * 70)
    print(report_str)
    print(f"\nDistribusi prediksi: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    print(f"\nTersimpan di: {args.output_dir}/metrics.json + confusion_matrix.png")

    # Peringatan kelas lemah (F1 < 0.5)
    weak = [c for c in class_names
            if report_dict.get(c, {}).get("f1-score", 1.0) < 0.5]
    if weak:
        print(f"\nPerhatian: kelas dengan F1 < 0.5: {', '.join(weak)}")
        print("Pertimbangkan tambah data / augmentasi / class weights saat training.")


if __name__ == "__main__":
    main()
