"""
Script untuk melatih model CNN untuk deteksi ekspresi wajah
Optimized version dengan class weights untuk handle imbalanced dataset
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modul lokal
import config
from model import create_expression_model, create_transfer_model, compile_model
from preprocessing import create_data_generators, print_dataset_info


def plot_training_history(history, save_path='training_history.png'):
    """
    Visualisasi hasil pelatihan (akurasi dan loss)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Akurasi
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot training history disimpan di: {save_path}")
    plt.close()


def compute_class_weights_from_generator(generator):
    """
    Hitung class weights untuk mengatasi imbalanced dataset
    """
    print("\n🔍 Menghitung class weights untuk balanced training...")
    
    # Get class indices dan counts
    class_counts = {}
    for class_name, class_idx in generator.class_indices.items():
        class_dir = os.path.join(generator.directory, class_name)
        count = len([f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_idx] = count
    
    # Print class distribution
    print("\n📊 Distribusi Dataset:")
    print("-" * 50)
    total_samples = sum(class_counts.values())
    for class_name, class_idx in sorted(generator.class_indices.items(), key=lambda x: x[1]):
        count = class_counts[class_idx]
        percentage = (count / total_samples) * 100
        print(f"  {class_name:10s}: {count:5d} sampel ({percentage:5.2f}%)")
    print("-" * 50)
    print(f"  {'TOTAL':10s}: {total_samples:5d} sampel")
    
    # Compute class weights
    class_labels = []
    for class_idx in range(len(class_counts)):
        class_labels.extend([class_idx] * class_counts[class_idx])
    
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
    
    # Print class weights
    print("\n⚖️  Class Weights (untuk balancing):")
    print("-" * 50)
    for class_name, class_idx in sorted(generator.class_indices.items(), key=lambda x: x[1]):
        weight = class_weights_dict[class_idx]
        print(f"  {class_name:10s}: {weight:6.3f}")
    print("-" * 50)
    print("💡 Kelas minoritas mendapat weight lebih tinggi\n")
    
    return class_weights_dict


def train_model(train_dir, validation_dir, 
                epochs=100,
                batch_size=64, 
                learning_rate=0.0001,
                input_shape=config.INPUT_SHAPE,
                model_save_path=config.MODEL_PATH,
                use_class_weights=True):
    """
    Melatih model CNN untuk deteksi ekspresi wajah
    """
    
    print("\n" + "=" * 70)
    print("TRAINING MODEL DETEKSI EKSPRESI WAJAH")
    print("=" * 70)
    print(f"⚙️  Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Class Weights: {'Enabled' if use_class_weights else 'Disabled'}")
    print("=" * 70)
    
    # 1. Buat Data Generators
    print("\n[1/6] Membuat data generators dengan augmentasi...")
    train_generator, validation_generator = create_data_generators(
        train_dir, 
        validation_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        augmentation=True
    )
    
    print_dataset_info(train_generator, validation_generator)
    
    # 2. Compute Class Weights
    class_weights = None
    if use_class_weights:
        print("\n[2/6] Computing class weights untuk balanced training...")
        class_weights = compute_class_weights_from_generator(train_generator)
    else:
        print("\n[2/6] Class weights disabled, menggunakan equal weights...")
        class_weights = None
    
    # 3. Buat Model
    print("\n[3/6] Membuat arsitektur model CNN...")
    num_classes = train_generator.num_classes
    
    if config.USE_TRANSFER_LEARNING:
        print("   Using Transfer Learning (MobileNetV2)")
        model = create_transfer_model(input_shape=input_shape, num_classes=num_classes)
    else:
        print("   Using Custom CNN (Baseline)")
        model = create_expression_model(input_shape=input_shape, num_classes=num_classes)
        
    model = compile_model(model, learning_rate=learning_rate)
    
    print(f"\n📋 Model Summary:")
    model.summary()
    print(f"\n💾 Total Parameters: {model.count_params():,}")
    
    # 4. Setup Callbacks
    print("\n[4/6] Setup callbacks untuk pelatihan...")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger('training_log.csv', append=False)
    ]
    
    # 5. Training
    print(f"\n[5/6] Mulai training untuk {epochs} epochs...")
    print("=" * 70)
    print()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 6. Evaluasi dan Simpan Hasil
    print("\n[6/6] Evaluasi dan simpan hasil...")
    
    plot_training_history(history, 'training_history.png')
    
    # Simpan class labels
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    
    with open(config.LABELS_PATH, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"✓ Class labels disimpan di: {config.LABELS_PATH}")
    
    # Evaluasi final
    print("\n" + "=" * 70)
    print("HASIL TRAINING")
    print("=" * 70)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"Train Accuracy: {final_train_acc:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Train Loss: {final_train_loss:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"\nModel disimpan di: {model_save_path}")
    print("=" * 70)
    
    return model, history


if __name__ == '__main__':
    # Konfigurasi training
    training_config = {
        'train_dir': config.TRAIN_DIR,
        'validation_dir': config.VALIDATION_DIR,
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'input_shape': config.INPUT_SHAPE,
        'model_save_path': config.MODEL_PATH,
        'use_class_weights': True
    }
    
    # Cek apakah dataset ada
    if not os.path.exists(training_config['train_dir']):
        print(f"Error: Folder {training_config['train_dir']} tidak ditemukan!")
        print("Silakan tambahkan data training ke folder tersebut terlebih dahulu.")
        sys.exit(1)
    
    if not os.path.exists(training_config['validation_dir']):
        print(f"Error: Folder {training_config['validation_dir']} tidak ditemukan!")
        print("Silakan tambahkan data validation ke folder tersebut terlebih dahulu.")
        sys.exit(1)
    
    # Mulai training
    try:
        model, history = train_model(**training_config)
        print("\n✓ Training selesai dengan sukses!")
    except Exception as e:
        print(f"\n✗ Error saat training: {e}")
        import traceback
        traceback.print_exc()
